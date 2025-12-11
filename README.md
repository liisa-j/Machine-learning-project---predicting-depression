# Machine-learning-project---predicting-depression
This is a repo for 'Machine learning' project, autumn 2025


This repository consists of: 
/.../


---

# **I Dataset**

The data used in this project is from: https://zenodo.org/records/5854911

Citation: Suhavi, Singh, A., Arora, U., Shrivastava, S., Singh, A., Shah, R. R., & Kumaraguru, P. (2022). Twitter Self-reported Temporally-contextual Mental Health Diagnosis (Twitter-STMHD) Dataset (Version 1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5854911 


## **1. Getting the data**

The datasets are **not included** in this repository due to size (3+ GB). Please download datasets manually:

- **Positive class (depressed users):** [depression.zip](https://zenodo.org/records/5854911/files/depression.zip?download=1)  
- **Negative class (control users):** [neg.zip](https://zenodo.org/records/5854911/files/neg.zip?download=1)  


In your repo root, create a data/ folder:
```bash
mkdir data
```

Unzip these depression.zip and neg.zip files into your project folder:

## Positive class
```bash
unzip depression.zip -d data/depression
````

## Negative class
```bash
unzip neg.zip -d data/neg
```

The file structure should look like this: 

```swift
data/
├─ depression/
│  └─ media/data_dump/udit/submission/dataset/depression/cleaned_data_depression/
├─ neg/
│  └─ media/data_dump/udit/submission/dataset/neg/cleaned_data_neg/
```

## **2. Scripts for data preprocessing**
*NB - If you for any reason want to rerun these scripts, please understand that running these scripts takes times, as does downloading the zipped datasets you need for running them. Thanks!*

**process_posclass.py** — Processes the positive (depressed) class.

Reads JSON files in depression_data/...
Flattens tweets, filters by ±90 days around the anchor tweet, and labels them
Outputs: data/tweets.parquet

**process_negclass.py** — processes the negative/control class.

Reads JSON files in neg_data/...
Flattens tweets in batches and labels them as control
Outputs: data/neg_combined.parquet


**final_df.py** - creates a final dataframe of 20 million lines (combing previously produced negative and positive class datasets)

Loads datasets using pandas.read_parquet.
Converts user_id to string to ensure consistent data types.
Combines datasets with pd.concat.
Shuffles the combined df randomly to remove ordering effects.
Saves the final combined df to Parquet. 

**short_df.py** - creates a smaller (1 million rows), randomly sampled version of the previous combined_tweets.parquet. 

Loads the combined df using pandas.read_parquet.
Randomly samples rows.
Shuffles df to remove any residual ordering.
Saves the sampled dataset to Parquet. 

## **3. Scripts for feature extraction**

Here will be the explanations about extracting features (LIWC, textstat, stylo etc... )


# **II Requirements**

Install the required Python packages:

```` bash
pip install -r requirements.txt
````

# **III Models**

Then next.... 
