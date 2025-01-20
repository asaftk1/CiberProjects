# README: Model Training and Inference

## Overview
This repository contains two main scripts:

1. `train.py` - Handles model training and saves the test dataset and models.
2. `inference.py` - Performs inference using the trained model, providing predictions and accuracy metrics.

Follow the instructions below to run these scripts and evaluate the model's performance.

---

## Prerequisites

### System Requirements
- Python 3.8 or above
- GPU (Optional but recommended for training)


### Required Files
download the dataset from this link : https://drive.google.com/file/d/1gc6a5tT2Pmb-OIbDpZjSfjW_qkNYk7Pm/view?usp=sharing
Ensure the training dataset is added to the folder.(the dataset is in the link added to the submition)
 The test dataset for inference will be the one saved by the training script (`test_dataset.csv`).
---

## How to Run

### Step 1: Train the Model
Run the `train.py` script to train the model and save required files for inference.


#### What Happens During Training:
1. The script loads and preprocesses the dataset.
2. It extracts features using TF-IDF and additional custom features.
3. A Random Forest classifier and fine-tuned RoBERTa model are trained.
4. Stacking is applied using a Logistic Regression meta-classifier.
5. The following artifacts are saved:
   - `random_forest_model.pkl`: Random Forest model
   - `stacking_meta_classifier.pkl`: Meta-classifier for stacking
   - `tfidf_vectorizer.pkl`: TF-IDF vectorizer
   - `feature_metadata.pkl`: Metadata about features
   - `fine_tuned_roberta`: Fine-tuned RoBERTa model and tokenizer
   - `test_dataset.csv`: Processed test dataset

### Step 2: Perform Inference
After training, run the `inference.py` script to generate predictions and evaluate the model.


#### What Happens During Inference:
1. The script loads the saved models and metadata.
2. It preprocesses the test dataset to match the training features.
3. Predictions are generated for each email in the dataset.
4. Accuracy and classification metrics are computed if true labels are available.

---

## Output

### Training Output
- Logs with feature information and training progress
- Saved artifacts as described above

### Inference Output
- Predictions appended to the test dataset (`Prediction` column)
- Accuracy and classification report printed to the console (if labels are provided)

---

## Notes
- Ensure all required paths are correct in both scripts before running.
- GPU acceleration is highly recommended for training RoBERTa.
