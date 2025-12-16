# How to Run the Depression Prediction Project

## ✅ Setup Complete

All dependencies have been installed:
- ✅ Python packages from requirements.txt
- ✅ spaCy English model (en_core_web_sm)
- ✅ vaderSentiment (added to requirements)
- ✅ Data directory created

## 📋 Required Data Files

The project requires Twitter datasets that must be downloaded manually (3+ GB total):

1. **Download the datasets:**
   - Positive class (depressed users): https://zenodo.org/records/5854911/files/depression.zip?download=1
   - Negative class (control users): https://zenodo.org/records/5854911/files/neg.zip?download=1

2. **Extract the data:**
   ```bash
   unzip depression.zip -d data/depression
   unzip neg.zip -d data/neg
   ```

   Expected structure:
   ```
   data/
   ├─ depression/
   │  └─ media/data_dump/udit/submission/dataset/depression/cleaned_data_depression/
   ├─ neg/
   │  └─ media/data_dump/udit/submission/dataset/neg/cleaned_data_neg/
   ```

## 🔄 Complete Workflow

### Step 1: Process Raw Data
```bash
# Process positive class (depressed users)
python src/process_posclass.py

# Process negative class (control users)
python src/process_negclass.py
```

### Step 2: Combine Datasets
```bash
# Create combined dataset (20M rows)
python src/final_df.py

# Create smaller sample (1M rows)
python src/short_df.py
```

### Step 3: Preprocess Data
```bash
# Clean the tweets
python src/preprocessing.py
```

### Step 4: Extract Features
```bash
# Extract linguistic and sentiment features
python src/features_shorter2.py
```

### Step 5: Train Models
```bash
# Random Forest
python src/RF.py

# Logistic Regression
python src/linear_regression.py

# SVM
python src/svm.py
```

## 🚀 Quick Start (If You Have Processed Data)

If you already have `data/shorty_features.parquet`, you can directly run any model:

```bash
python src/linear_regression.py
python src/RF.py
python src/svm.py
```

## 📊 Expected Output

Each model script will output:
- Classification Report (precision, recall, F1-score)
- Confusion Matrix
- Training time (for RF.py)

## ⚠️ Current Status

**Missing:** Raw data files need to be downloaded and extracted before running the preprocessing pipeline.

**Ready:** All code and dependencies are set up and ready to run once data is available.


