## This script takes two downloaded Reddit csv files and
## outputs 2 parquet files: 
## 1) reddit_full_dataset.parquet - this is identical to 
## the dataset we used in our intermediate presentation
## 2) reddit_text_classification_dataset.parquet - this dataset has 
## only 3 columns (author, clean_text, label_encoded)


from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


depression_file = DATA_DIR / "depression_pre_features_tfidf_256.csv"
fitness_file = DATA_DIR / "fitness_pre_features_tfidf_256.csv"

# Load data and combine
df_depression = pd.read_csv(depression_file, sep=",")
df_fitness = pd.read_csv(fitness_file, sep=",")
df_combined = pd.concat([df_depression, df_fitness], ignore_index=True)

# Label: 1 = depressed, 0 = not depressed
df_combined['depressed'] = df_combined['subreddit'].map({'depression': 1, 'fitness': 0})
df_combined = df_combined.drop(columns=['subreddit'])

# Drop TF-IDF columns
tfidf_cols = df_combined.filter(regex=r"^tfidf").columns
df_combined = df_combined.drop(columns=tfidf_cols)

# Add feature: absolutist word count
absolutist_words = ["always", "never", "entire", "totally"]

def count_absolutist_words(text, words_list):
    words = re.findall(r'\b\w+\b', str(text).lower())
    return sum(w in words_list for w in words)

df_combined["absolutist"] = df_combined["post"].apply(lambda x: count_absolutist_words(x, absolutist_words))

# Feature: % absolutist words
def absolutist_percentage(text, n_words, words_list):
    if n_words == 0:
        return 0
    words = str(text).lower().split()
    count = sum(1 for w in words if w in words_list)
    return (count / n_words) * 100

df_combined["absolutist_pct"] = df_combined.apply(
    lambda row: absolutist_percentage(row["post"], row["n_words"], absolutist_words), axis=1
)

# Shuffle dataset
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)


# 1. Full dataset to parquet
full_parquet_path = DATA_DIR / "reddit_full_dataset.parquet"
df_combined.to_parquet(full_parquet_path, index=False)
print(f"Saved full dataset to: {full_parquet_path}")


# 2. Short dataset ready for feature extraction
TEXT_COL = "post"
LABEL_COL = "depressed"
USER_COL = "author"

df_ml = df_combined[[USER_COL, TEXT_COL, LABEL_COL]].copy()

# Rename text column to match feature extraction input
df_ml = df_ml.rename(columns={TEXT_COL: "clean_text"})

# Encode labels as integers for feature extraction
le = LabelEncoder()
df_ml["label_encoded"] = le.fit_transform(df_ml[LABEL_COL])


df_ml = df_ml.drop(columns=[LABEL_COL])
df_ml = df_ml.dropna(subset=["clean_text"])

# Save parquet
ml_parquet_path = DATA_DIR / "reddit_text_classification_dataset.parquet"
df_ml.to_parquet(ml_parquet_path, index=False)
print(f"Saved ML-ready dataset to: {ml_parquet_path}")