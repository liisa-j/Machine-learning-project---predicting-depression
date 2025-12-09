# Machine-learning-project---predicting-depression
This is a repo for 'Machine learning' project, autumn 2025



This project processes Twitter data to create datasets for predicting depression. It includes scripts to process both the **positive class (depressed users)** and the **negative/control class**.

---

# **1. Dataset**

The datasets are **not included** in this repository due to size (3+ GB). You need to download them manually:

- **Positive class (depressed users):** [depression.zip](https://zenodo.org/records/5854911/files/depression.zip?download=1)  
- **Negative class (control users):** [neg.zip](https://zenodo.org/records/5854911/files/neg.zip?download=1)  



In your repo root, create a data/ folder:
```bash
mkdir data
```

Unzip these depression.zip and neg.zip files into your project folder:

## Positive class
```bash
unzip depression.zip -d data/depression_data
````

## Negative class
```bash
unzip neg.zip -d data/neg_data
```

The file structure should look like this: 

```swift
data/
├─ depression_data/
│  └─ media/data_dump/udit/submission/dataset/depression/cleaned_data_depression/
├─ neg_data/
│  └─ media/data_dump/udit/submission/dataset/neg/cleaned_data_neg/
```

# **2. Scripts**

process_posclass.py — Processes the positive (depressed) class.

Reads JSON files in depression_data/...

Flattens tweets, filters by ±90 days around the anchor tweet, and labels them

Outputs: data/tweets.parquet

process_negclass.py — Processes the negative/control class.

Reads JSON files in neg_data/...

Flattens tweets in batches and labels them as control

Outputs: data/neg_combined.parquet

Note: Both scripts expect the folder structure above. If you unzip the datasets elsewhere, update DATA_ROOT and neg_root in the scripts accordingly.


# **3. Requirements**

Install the required Python packages:

```` bash
pip install -r requirements.txt
````

# **4. .... **

Then next.... 