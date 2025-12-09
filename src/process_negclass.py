import os
import json
import pandas as pd
from tqdm import tqdm
from dateutil import parser
import pytz
import random


NEG_ROOT = "data/neg_data/media/data_dump/udit/submission/dataset/neg/cleaned_data_neg"
OUTPUT_FILE = "data/neg_sampled.parquet"

# Number of negative tweets to sample = number of positive-class tweets
N_POSITIVE = 12151305 

# === HELPERS ===
def make_aware(dt_str):
    """Convert a datetime string to UTC-aware ISO 8601 format."""
    if not dt_str:
        return None
    dt = parser.isoparse(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.UTC)
    else:
        dt = dt.astimezone(pytz.UTC)
    return dt.isoformat()

def extract_user_tweets(user_folder):
    tweets_path = os.path.join(user_folder, "tweets.json")
    if not os.path.exists(tweets_path):
        return []

    try:
        with open(tweets_path, "r", encoding="utf-8") as f:
            tweets_dict = json.load(f)
    except:
        return []

    rows = []
    for date_str, tweets_list in tweets_dict.items():
        for t in tweets_list:
            rows.append({
                "user_id": os.path.basename(user_folder),
                "tweet_text": t.get("text"),
                "tweet_date": make_aware(t.get("timestamp_tweet")),
                "is_anchor": False,
                "label": "control"
            })
    return rows

# === MAIN ===
all_neg_rows = []
user_folders = [os.path.join(NEG_ROOT, d) for d in os.listdir(NEG_ROOT) if os.path.isdir(os.path.join(NEG_ROOT, d))]

for folder in tqdm(user_folders, desc="Processing negative users"):
    all_neg_rows.extend(extract_user_tweets(folder))

print(f"Total negative tweets before sampling: {len(all_neg_rows)}")

# Randomly sample negative tweets to match positive class size
if len(all_neg_rows) > N_POSITIVE:
    all_neg_rows = random.sample(all_neg_rows, N_POSITIVE)

# Convert to DataFrame and save
df_neg = pd.DataFrame(all_neg_rows)
df_neg.to_parquet(OUTPUT_FILE, engine="pyarrow", index=False)
print(f"Saved {df_neg.shape[0]} negative tweets to {OUTPUT_FILE}")