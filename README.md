# Machine-learning-project---predicting-depression
This is a repo for 'Machine learning' project, autumn 2025



This project processes Twitter data to create datasets for predicting depression. It includes scripts to process both the **positive class (depressed users)** and the **negative/control class**.

---

## **1. Dataset**

The datasets are **not included** in this repository due to size (3+ GB). You need to download them manually:

- **Positive class (depressed users):** [depression.zip](https://zenodo.org/records/5854911/files/depression.zip?download=1)  
- **Negative class (control users):** [neg.zip](https://zenodo.org/records/5854911/files/neg.zip?download=1)  



In your repo root, create a data/ folder:
```bash
mkdir data
```

Unzip these depression.zip and neg.zip files into your project folder:

# Positive class
```bash
unzip depression.zip -d data/depression_data
````

# Negative class
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

