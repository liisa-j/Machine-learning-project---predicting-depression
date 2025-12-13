from pathlib import Path
import pandas as pd
import numpy as np
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Input files
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

# Droo cols
cols_to_drop = [
    'economic_stress_total',
    'isolation_total',
    'substance_use_total',
    'guns_total',
    'domestic_stress_total',
    'suicidality_total',
    'author',
    'date',
    'post',
    'liwc_achievement', 'liwc_biological',
    'liwc_body', 'liwc_death','liwc_family',
    'liwc_friends', 'liwc_health',  'liwc_home',
    'liwc_humans', 'liwc_ingestion',  'liwc_leisure',
    'liwc_money', 'liwc_motion', 'liwc_religion',
    'liwc_sexual', 'liwc_work',
]

df_model = df_combined.drop(columns=cols_to_drop)

# 1. Full dataset to parquet
full_parquet_path = DATA_DIR / "full_dataset.parquet"
df_combined.to_parquet(full_parquet_path, index=False)
print(f"Saved full dataset to: {full_parquet_path}")

# 2. Short dataset (user_id, text, label) ready to parqet
TEXT_COL = "post"          
LABEL_COL = "depressed"    
USER_COL = "author"        

df_ml = df_combined[[USER_COL, TEXT_COL, LABEL_COL]].copy()
df_ml[LABEL_COL] = df_ml[LABEL_COL].map({1: "depressed", 0: "not_depressed"})
df_ml = df_ml.dropna(subset=[TEXT_COL, LABEL_COL])

ml_parquet_path = DATA_DIR / "text_classification_dataset.parquet"
df_ml.to_parquet(ml_parquet_path, index=False)
print(f"Saved ML-ready dataset to: {ml_parquet_path}")