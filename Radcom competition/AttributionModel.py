import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
train_file_path = 'train.csv'
test_file_path = 'test.csv'

print("Loading datasets...")
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Drop irrelevant columns
irrelevant_columns = ['Source_IP', 'Source_port', 'Destination_IP', 'Destination_port', 'Timestamp']
train_data_clean = train_data.drop(columns=irrelevant_columns, errors='ignore')
test_data_clean = test_data.drop(columns=irrelevant_columns, errors='ignore')

# Add derived features
train_data_clean['total_packet_length'] = train_data_clean['fwd_packets_length'] + train_data_clean['bwd_packets_length']
train_data_clean['average_packet_size'] = train_data_clean['total_packet_length'] / (train_data_clean['fwd_packets_amount'] + train_data_clean['bwd_packets_amount'] + 1e-5)
train_data_clean['packet_rate'] = (train_data_clean['fwd_packets_amount'] + train_data_clean['bwd_packets_amount']) / (train_data_clean['silence_windows'] + 1e-5)

test_data_clean['total_packet_length'] = test_data_clean['fwd_packets_length'] + test_data_clean['bwd_packets_length']
test_data_clean['average_packet_size'] = test_data_clean['total_packet_length'] / (test_data_clean['fwd_packets_amount'] + test_data_clean['bwd_packets_amount'] + 1e-5)
test_data_clean['packet_rate'] = (test_data_clean['fwd_packets_amount'] + test_data_clean['bwd_packets_amount']) / (test_data_clean['silence_windows'] + 1e-5)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in ['Protocol', 'attribution']:
    train_data_clean[column] = label_encoder.fit_transform(train_data_clean[column])
    test_data_clean[column] = label_encoder.transform(test_data_clean[column])

# Directly use the provided feature list for selection
selected_features_list = [
    'Protocol', 'fwd_packets_amount', 'bwd_packets_amount', 'fwd_packets_length', 'bwd_packets_length', 'max_fwd_packet ',
    'min_fwd_packet', 'max_bwd_packet', 'min_bwd_packet', 'FIN_count', 'SYN_count', 'RST_count', 'PSH_count',
    'min_fwd_inter_arrival_time', 'max_fwd_inter_arrival_time', 'mean_fwd_inter_arrival_time', 'max_bwd_inter_arrival_time',
    'mean_bwd_inter_arrival_time', 'max_bib_inter_arrival_time', 'mean_bib_inter_arrival_time',
    'min_packet_size', 'max_packet_size', 'mean_packet_size', 'STD_packet_size', 'mean_delta_byte', 'STD_delta_byte',
    'bandwidth_1', 'bandwidth_2', 'bandwidth_3', 'bandwidth_5', 'pps_fwd', 'pps_bwd', 'count_big_requests',
    'ACK_count', 'total_packet_length', 'average_packet_size', 'packet_rate'
]

# Separate features and target variable
X_train = train_data_clean[selected_features_list]
y_train = train_data_clean['attribution']
X_test = test_data_clean[selected_features_list]
y_test = test_data_clean['attribution']

# Train a Random Forest Classifier with optimized parameters
print("\nTraining Random Forest Classifier with Optimized Parameters...")
rf_model = RandomForestClassifier(
    class_weight={0: 1, 1: 1, 2: 2, 3: 1, 4: 2},  # Penalize misclassifications for audio (2) and video (4)
    random_state=42,
    n_estimators=300,        # More trees for stability
    max_depth=None,          # Allow trees to grow fully
    min_samples_leaf=1,      # Minimum samples per leaf node
    min_samples_split=2,     # Reduce split size
    n_jobs=-1
)

# Evaluate with Stratified Cross-Validation
print("\nPerforming Stratified Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring='accuracy')
print("\nStratified Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Stratified Cross-Validation Accuracy:", cv_scores.mean())

# Train and Evaluate on Test Set
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf, average='weighted')
recall = recall_score(y_test, y_pred_rf, average='weighted')
f1 = f1_score(y_test, y_pred_rf, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred_rf)
class_report = classification_report(y_test, y_pred_rf)

# Output metrics
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Display top features
feature_importances = pd.DataFrame({
    'Feature': selected_features_list,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)


# Load the validation dataset (inference only)
validation_file_path = 'val_without_labels.csv'
print("\nLoading validation dataset...")
validation_data = pd.read_csv(validation_file_path)

# Drop irrelevant columns
validation_data_clean = validation_data.drop(columns=irrelevant_columns, errors='ignore')

# Add derived features
validation_data_clean['total_packet_length'] = validation_data_clean['fwd_packets_length'] + validation_data_clean['bwd_packets_length']
validation_data_clean['average_packet_size'] = validation_data_clean['total_packet_length'] / (validation_data_clean['fwd_packets_amount'] + validation_data_clean['bwd_packets_amount'] + 1e-5)
validation_data_clean['packet_rate'] = (validation_data_clean['fwd_packets_amount'] + validation_data_clean['bwd_packets_amount']) / (validation_data_clean['silence_windows'] + 1e-5)

# Handle unseen labels in the Protocol column by assigning a new label for unseen values
unseen_label = len(label_encoder.classes_)
validation_data_clean['Protocol'] = validation_data_clean['Protocol'].apply(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else unseen_label)

# Separate features for validation
X_validation = validation_data_clean[selected_features_list]

# Make predictions on the validation dataset
y_pred_validation = rf_model.predict(X_validation)

# Decode predictions back to original labels
y_pred_validation_decoded = label_encoder.inverse_transform(y_pred_validation)

# Save validation predictions to a CSV file
validation_predictions = pd.DataFrame({'prediction': y_pred_validation_decoded})
output_file_path = 'validation_predictions_inference.csv'
validation_predictions.to_csv(output_file_path, index=False)
print(f"\nValidation predictions saved to {output_file_path}")
