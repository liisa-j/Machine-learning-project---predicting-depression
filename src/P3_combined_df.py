## Preprocessing: 3 ##
## This code takes in the neg_sampled_english.parquet (negative class) 
## and tweets_english.parquet (positive class) and returns combined_tweets.parquet
## 24 million rows shuffled dataset

import pandas as pd
from tqdm import tqdm


neg_file = "../data/neg_sampled_english.parquet"
dep_file = "../data/tweets_english.parquet"
output_file = "../data/combined_tweets.parquet"

print("Loading negative tweets...")
df_neg = pd.read_parquet(neg_file, engine="pyarrow")
print("Loading depressed tweets...")
df_dep = pd.read_parquet(dep_file, engine="pyarrow")

# types
df_neg['user_id'] = df_neg['user_id'].astype(str)
df_dep['user_id'] = df_dep['user_id'].astype(str)

print("Combining datasets...")
df_combined = pd.concat([df_neg, df_dep], ignore_index=True)

# shuffle
print("Shuffling...")
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# save as parquet
print(f"Saving combined dataset to {output_file}...")
df_combined.to_parquet(output_file, engine="pyarrow", index=False)
print(f"Done! Total rows: {df_combined.shape[0]}")