import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours as ENN
import pandas as pd

def add_features(data):
    # Normalize column names (strip whitespace)
    data.columns = data.columns.str.strip()

    # Ensure required columns exist before operations
    required_columns = [
        'fwd_packets_amount', 'bwd_packets_amount',
        'fwd_packets_length', 'bwd_packets_length',
        'pps_fwd', 'pps_bwd', 'ACK_count',
        'Protocol', 'Destination_port', 'Source_port',
        'Timestamp', 'Source_IP', 'Destination_IP'
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return data  # Return the data as-is if columns are missing
    else:
        # 1. Packet Ratios (before dropping the original columns)
        data['bwd_to_fwd_packets_ratio'] = data['bwd_packets_amount'] / (data['fwd_packets_amount'] + 1e-6)
        data['bwd_to_fwd_length_ratio'] = data['bwd_packets_length'] / (data['fwd_packets_length'] + 1e-6)

        # 2. Total Traffic (before dropping the original columns)
        data['total_packets'] = data['fwd_packets_amount'] + data['bwd_packets_amount']

        # 3. Activity Intensity (combined features)
        data['activity_intensity'] = np.log1p(data['pps_fwd']) + np.log1p(data['pps_bwd']) + np.log1p(data['ACK_count'])

        # 4. Temporal Features
        data['hour_of_day'] = pd.to_datetime(data['Timestamp'], unit='s').dt.hour

        # 7. Beacon Aggregation (Re-added as before)
        beacon_columns = [col for col in data.columns if col.startswith('beaconning_')]
        if beacon_columns:
            data['beaconning_score'] = data[beacon_columns].sum(axis=1)

        # 8. Traffic Distribution Features (Ratios and Differences)
        data['packet_size_ratio'] = data['fwd_packets_length'] / (data['bwd_packets_length'] + 1e-6)
        data['total_packet_ratio'] = data['fwd_packets_amount'] / (data['bwd_packets_amount'] + 1e-6)

        # 9. Activity Ratios (between traffic in different directions)
        data['activity_ratio'] = data['pps_fwd'] / (data['pps_bwd'] + 1e-6)

        # 10. Aggregated Length Features
        data['mean_packet_size'] = data['total_packets'] / data['total_packets']
        data['std_packet_size'] = data[['fwd_packets_length', 'bwd_packets_length']].std(axis=1)

        # 11. Hour of Day Features (time-based)
        data['peak_hour'] = data['hour_of_day'].apply(lambda x: 1 if x >= 8 and x <= 18 else 0)

        # Add first packet size features (from the previous code)
        first_packet_columns = [f'first_packet_sizes_{i}' for i in range(30)]

        # Mean, Std, Min, Max, Skewness, Kurtosis of the first 30 packet sizes
        data['mean_first_packet_size'] = data[first_packet_columns].mean(axis=1)
        data['std_first_packet_size'] = data[first_packet_columns].std(axis=1)
        data['min_first_packet_size'] = data[first_packet_columns].min(axis=1)
        data['max_first_packet_size'] = data[first_packet_columns].max(axis=1)

        # Skewness and Kurtosis
        data['skew_first_packet_size'] = data[first_packet_columns].apply(lambda x: pd.Series(x).skew(), axis=1)
        data['kurt_first_packet_size'] = data[first_packet_columns].apply(lambda x: pd.Series(x).kurtosis(), axis=1)

        # Sum of first 30 packet sizes
        data['sum_first_packet_size'] = data[first_packet_columns].sum(axis=1)

        # Count of small (<1500 bytes) and large (>1500 bytes) packets
        data['count_small_packets'] = data[first_packet_columns].apply(lambda x: np.sum(np.array(x) < 1500), axis=1)
        data['count_large_packets'] = data[first_packet_columns].apply(lambda x: np.sum(np.array(x) > 1500), axis=1)

        # Percentiles of first 30 packet sizes
        data['percentile_25'] = data[first_packet_columns].apply(lambda x: np.percentile(x, 25), axis=1)
        data['percentile_50'] = data[first_packet_columns].apply(lambda x: np.percentile(x, 50), axis=1)
        data['percentile_75'] = data[first_packet_columns].apply(lambda x: np.percentile(x, 75), axis=1)

        # Drop the old columns after creating new features
        data.drop(['fwd_packets_amount', 'bwd_packets_amount', 'fwd_packets_length', 'bwd_packets_length',
                   'pps_fwd', 'pps_bwd', 'ACK_count', 'Timestamp', 'Protocol', 'Destination_port',
                   'Source_port', 'Source_IP', 'Destination_IP'] + beacon_columns + first_packet_columns, axis=1,
                  inplace=True)

        return data


# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data = add_features(train_data)
test_data = add_features(test_data)

# Full set of selected features (after preprocessing)
selected_features = [
    'max_fwd_packet', 'min_fwd_packet', 'max_bwd_packet', 'min_bwd_packet',
    'FIN_count', 'SYN_count', 'RST_count', 'PSH_count', 'silence_windows',
    'min_fwd_inter_arrival_time', 'max_fwd_inter_arrival_time', 'mean_fwd_inter_arrival_time',
    'min_bwd_inter_arrival_time', 'max_bwd_inter_arrival_time', 'mean_bwd_inter_arrival_time',
    'min_bib_inter_arrival_time', 'max_bib_inter_arrival_time', 'mean_bib_inter_arrival_time',
    'min_packet_size', 'max_packet_size', 'mean_packet_size', 'STD_packet_size',
    'mean_delta_byte', 'STD_delta_byte', 'count_big_requests',
    'label', 'bwd_to_fwd_packets_ratio', 'bwd_to_fwd_length_ratio', 'total_packets',
    'activity_intensity', 'hour_of_day', 'beaconning_score', 'packet_size_ratio',
    'total_packet_ratio', 'activity_ratio', 'std_packet_size', 'peak_hour',
    'mean_first_packet_size', 'std_first_packet_size', 'min_first_packet_size',
    'max_first_packet_size', 'skew_first_packet_size', 'kurt_first_packet_size',
    'sum_first_packet_size', 'count_small_packets', 'count_large_packets',
    'percentile_25', 'percentile_50', 'percentile_75'
]

# Preprocess: Filter the columns and convert 'label' to numeric
train_data_filtered = train_data[selected_features].copy()
test_data_filtered = test_data[selected_features].copy()

# Convert categorical 'label' to numeric codes
train_data_filtered['label'] = train_data_filtered['label'].astype('category').cat.codes
test_data_filtered['label'] = test_data_filtered['label'].astype('category').cat.codes

# Split features and target
X_train = train_data_filtered.drop(columns=['label'])
y_train = train_data_filtered['label']
X_test = test_data_filtered.drop(columns=['label'])
y_test = test_data_filtered['label']

# Check for NaN or Inf values
if X_train.isnull().any().any() or X_train.isin([float('inf'), float('-inf')]).any().any():
    print("Warning: Found NaN or Inf values in training data.")
    X_train = X_train.fillna(0)  # Replace NaNs with 0 or appropriate value
    X_train = X_train.replace([float('inf'), float('-inf')], 0)  # Replace Inf values with 0
# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Apply ENN to clean up the data
enn = ENN()
X_train_cleaned, y_train_cleaned = enn.fit_resample(X_train_balanced, y_train_balanced)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],  # Reduced n_estimators to speed up
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced'],
    'max_features': ['sqrt', 'log2', None],  # Added max_features for tuning
    'bootstrap': [True, False]  # Added bootstrap parameter
}

print("Performing Grid Search...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,  # Using 3-fold cross-validation
    verbose=2,
    n_jobs=-1,
    scoring='accuracy'  # Explicitly specify the scoring metric
)
grid_search.fit(X_train_balanced, y_train_balanced)
print("Grid Search Completed.")
best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)

# Evaluate the model
print("Evaluating the model...")
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred, zero_division=0)
print("\nClassification Report:\n", report)

# Load additional data for validation
validation_data_path = 'val_without_labels.csv'
validation_data = pd.read_csv(validation_data_path)
validation_data = add_features(validation_data)

# Adjust selected features for validation (exclude 'label')
selected_features_no_label = [feature for feature in selected_features if feature != 'label']
X_validation = validation_data[selected_features_no_label]

# Predict on validation data
y_validation_pred = best_model.predict(X_validation)

# Decode labels if needed
label_decoder = dict(enumerate(train_data['label'].astype('category').cat.categories))
y_validation_pred_decoded = [label_decoder[label] for label in y_validation_pred]

# Output predictions to CSV
output_path = 'predicted_labels.csv'
validation_output = pd.DataFrame({
    'Predicted_Label': y_validation_pred_decoded
})
validation_output.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")
