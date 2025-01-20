import pandas as pd
import joblib
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

# Load the saved models and vectorizer
meta_classifier = joblib.load('stacking_meta_classifier.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
roberta_model = RobertaForSequenceClassification.from_pretrained('fine_tuned_roberta')
roberta_tokenizer = RobertaTokenizer.from_pretrained('fine_tuned_roberta')

# Load saved metadata for feature consistency
feature_metadata = joblib.load('feature_metadata.pkl')
expected_tfidf_features = len(tfidf.get_feature_names_out())  # Dynamically derive TF-IDF feature count
expected_additional_features = 5  # From training
roberta_embedding_dim = feature_metadata['roberta_dim']

# Additional feature extraction functions
def extract_additional_features(text):
    # Match training features
    text_length = len(text)
    special_characters = sum(1 for char in text if not char.isalnum() and not char.isspace())
    uppercase_letters = sum(1 for char in text if char.isupper())
    stop_word_ratio = len([word for word in text.split() if word.lower() in stop_words]) / max(1, len(text.split()))
    average_word_length = np.mean([len(word) for word in text.split()]) if text.split() else 0

    # Return only the features used during training
    return np.array([
        text_length, special_characters, uppercase_letters,
        stop_word_ratio, average_word_length
    ])

# Define stop words
stop_words = set(stopwords.words('english'))

# Function to preprocess text and combine features
def preprocess_text_with_features(text):
    # TF-IDF features
    tfidf_features = tfidf.transform([text]).toarray()
    if tfidf_features.shape[1] != expected_tfidf_features:
        raise ValueError(f"TF-IDF feature mismatch: Expected {expected_tfidf_features}, but got {tfidf_features.shape[1]}")

    # Additional features
    additional_features = extract_additional_features(text).reshape(1, -1)
    if additional_features.shape[1] != expected_additional_features:
        raise ValueError(f"Additional feature mismatch: Expected {expected_additional_features}, but got {additional_features.shape[1]}")

    # Combine all features
    combined_features = np.hstack([tfidf_features, additional_features])
    return combined_features

# Batch inference for faster processing
def predict_batch(texts):
    # Preprocess text features
    combined_features = np.vstack([preprocess_text_with_features(text) for text in texts])

    # Get RoBERTa logits for the batch
    inputs = roberta_tokenizer(
        texts,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    )
    roberta_features = roberta_model(**inputs).logits.detach().numpy()

    if roberta_features.shape[1] != roberta_embedding_dim:
        raise ValueError(f"RoBERTa embedding mismatch: Expected {roberta_embedding_dim}, but got {roberta_features.shape[1]}")

    # Combine features for final prediction
    final_features = np.hstack([combined_features, roberta_features])
    predictions = meta_classifier.predict(final_features)
    return ["Phishing Email" if pred == 1 else "Safe Email" for pred in predictions]

# Load the validation dataset
file_path = 'test_dataset.csv.csv'
test_dataset = pd.read_csv(file_path, low_memory=False)

# Check for missing or invalid data
if 'Email Text' not in test_dataset.columns:
    raise ValueError("The test dataset does not contain the required 'Email Text' column.")
test_dataset = test_dataset.dropna(subset=['Email Text'])

# Batch size for processing
batch_size = 32
predictions = []
for i in range(0, len(test_dataset), batch_size):
    batch_texts = test_dataset['Email Text'].iloc[i:i+batch_size].tolist()
    predictions.extend(predict_batch(batch_texts))

# Save predictions
test_dataset['Prediction'] = predictions

# If true labels are available, evaluate the model's performance
if 'Email Type' in test_dataset.columns:
    # Map predictions and labels to numerical format
    test_dataset['Prediction_Numerical'] = test_dataset['Prediction'].map({'Safe Email': 0, 'Phishing Email': 1})
    y_true = test_dataset['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
    y_pred = test_dataset['Prediction_Numerical']

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nEnsemble Model Accuracy: {accuracy:.2%}")

    # Generate a classification report
    report = classification_report(y_true, y_pred)
    print("\nEnsemble Model Classification Report:")
    print(report)
