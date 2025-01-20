import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.corpus import stopwords

# Load and preprocess the dataset
file_path = 'dataset.csv' 
dataset = pd.read_csv(file_path, low_memory=False)

# Data cleaning and preprocessing
if 'Email Text' in dataset.columns and 'Email Type' in dataset.columns:
    dataset = dataset[['Email Text', 'Email Type']]
else:
    raise ValueError("Dataset does not contain 'Email Text' or 'Email Type' columns.")

dataset = dataset.dropna(subset=['Email Text', 'Email Type'])
dataset['Text Length'] = dataset['Email Text'].apply(len)
dataset = dataset[(dataset['Text Length'] > 10) & (dataset['Text Length'] < 5000)]

# Feature engineering
def count_special_characters(text):
    return sum(1 for char in text if not char.isalnum() and not char.isspace())

def count_uppercase_letters(text):
    return sum(1 for char in text if char.isupper())

def calculate_stop_word_ratio(text, stop_words):
    words = text.split()
    stop_word_count = len([word for word in words if word.lower() in stop_words])
    return stop_word_count / max(1, len(words))

def average_word_length(text):
    words = text.split()
    return np.mean([len(word) for word in words]) if words else 0

stop_words = set(stopwords.words('english'))
dataset['Special Characters'] = dataset['Email Text'].apply(count_special_characters)
dataset['Uppercase Letters'] = dataset['Email Text'].apply(count_uppercase_letters)
dataset['Stop Word Ratio'] = dataset['Email Text'].apply(lambda x: calculate_stop_word_ratio(x, stop_words))
dataset['Average Word Length'] = dataset['Email Text'].apply(average_word_length)

# Target and text
y = dataset['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
X = dataset['Email Text']

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
X_tfidf = tfidf.fit_transform(X)

# Combine TF-IDF and additional features
additional_features = dataset[['Text Length', 'Special Characters', 'Uppercase Letters',
                               'Stop Word Ratio', 'Average Word Length']].values
X_combined = np.hstack((X_tfidf.toarray(), additional_features))

# Save feature metadata
joblib.dump({'n_features': X_combined.shape[1]}, 'feature_metadata.pkl')

# Train-test split for combined features
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

# Train-test split for raw text for RoBERTa
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Combine test data for saving
test_data = pd.DataFrame({
    'Email Text': X_test_text,
    'Email Type': y_test_text.map({0: 'Safe Email', 1: 'Phishing Email'})
})

# Save the test dataset to a CSV file
test_data.to_csv('test_dataset.csv', index=False)
print("Test dataset saved to 'test_dataset.csv'!")

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=200)
rf_model.fit(X_train, y_train)

# Save the Random Forest model
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Random Forest model saved!")

# Define a custom dataset for RoBERTa
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Fine-tune RoBERTa
max_len = 128
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_dataset = EmailDataset(pd.Series(X_train_text), pd.Series(y_train_text), tokenizer, max_len)
test_dataset = EmailDataset(pd.Series(X_test_text), pd.Series(y_test_text), tokenizer, max_len)

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    learning_rate=3e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    fp16=True,
)

# Define a function to compute evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Update Trainer to include the compute_metrics function
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # Include metrics computation
)

trainer.train()

# Save the fine-tuned RoBERTa model and tokenizer
model.save_pretrained('./fine_tuned_roberta')
tokenizer.save_pretrained('./fine_tuned_roberta')

print("Fine-tuned RoBERTa model and tokenizer saved!")

# Log the number of features during training
print(f"Number of TF-IDF features: {X_tfidf.shape[1]}")
print(f"Number of additional features: {additional_features.shape[1]}")
print(f"Total features in combined X_train: {X_combined.shape[1]}")

# Create embeddings from RoBERTa for stacking
train_embeddings = trainer.predict(train_dataset).predictions
test_embeddings = trainer.predict(test_dataset).predictions

# Combine TF-IDF features, additional features, and RoBERTa embeddings for stacking
X_train_stacking = np.hstack([X_train, train_embeddings])
X_test_stacking = np.hstack([X_test, test_embeddings])

# Train Logistic Regression as the meta-classifier
meta_classifier = LogisticRegression(random_state=42, max_iter=1000)
meta_classifier.fit(X_train_stacking, y_train)

# Save the stacking meta-classifier and feature metadata
joblib.dump(meta_classifier, 'stacking_meta_classifier.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump({'n_features': X_combined.shape[1], 'roberta_dim': train_embeddings.shape[1]}, 'feature_metadata.pkl')

print("Meta-classifier, TF-IDF vectorizer, and feature metadata saved!")
