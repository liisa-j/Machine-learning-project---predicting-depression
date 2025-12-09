import os
import json
import random
import pandas as pd

# Path to negative-class JSON data
NEG_ROOT = "data/neg_data/media/data_dump/udit/submission/dataset/neg/cleaned_data_neg"

# Number of tweets to sample (should match your positive class)
N_POSITIVE = 12151305

all_neg_rows = []

# Collect all negative tweets
user_folders = [os.path.join(NEG_ROOT, d) for d in os.listdir(NEG_ROOT) if os.path.isdir(os.path.join(NEG_ROOT, d))]

for folder in user_folders:
    tweets_file = os.path.join(folder, "tweets.json")
    if not os.path.exists(tweets_file):
        continue

    try:
        with open(tweets_file, "r", encoding="utf-8") as f:
            tweets_dict = json.load(f)
    except Exception as e:
        print(f"Skipping {folder} due to error: {e}")
        continue

    for tweet_list in tweets_dict.values():
        for t in tweet_list:
            tweet_text = t.get("text")
            tweet_date = t.get("timestamp_tweet")
            if not tweet_text or not tweet_date:
                continue

            all_neg_rows.append({
                "user_id": os.path.basename(folder),
                "tweet_text": tweet_text,
                "tweet_date": tweet_date,
                "is_anchor": False,
                "label": "control"
            })

# Random sample to match positive class size
if len(all_neg_rows) > N_POSITIVE:
    neg_sample = random.sample(all_neg_rows, N_POSITIVE)
else:
    neg_sample = all_neg_rows

# Convert to DataFrame and save as Parquet
df_neg_sample = pd.DataFrame(neg_sample)
output_path = "data/neg_sampled.parquet"
df_neg_sample.to_parquet(output_path, index=False, engine='pyarrow')

print(f"Saved {df_neg_sample.shape[0]} negative tweets to {output_path}")