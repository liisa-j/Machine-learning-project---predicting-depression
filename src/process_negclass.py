import os
import json
import pandas as pd
from tqdm import tqdm
from dateutil import parser
import pytz
import random

NEG_ROOT = "data/neg/media/data_dump/udit/submission/dataset/neg/cleaned_data_neg"
OUTPUT_FILE = "data/neg_sampled_english.parquet"

# Number of negative tweets to sample = number of positive tweets
N_POSITIVE = 10128861 

# === HELPERS ===
def make_aware(dt_str):
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
            tweet_lang = t.get("language")
            tweet_text = t.get("text")
            tweet_date = make_aware(t.get("timestamp_tweet"))

            # Only English tweets with text
            if tweet_text and tweet_lang == "en":
                rows.append({
                    "user_id": os.path.basename(user_folder),
                    "tweet_text": tweet_text,
                    "tweet_date": tweet_date,
                    "is_anchor": False,
                    "label": "control",
                    "language": "en"
                })
    return rows

# MEMORY-SAFE RESERVOIR SAMPLING 
sample = []
count = 0

user_folders = [os.path.join(NEG_ROOT, d)
                for d in os.listdir(NEG_ROOT)
                if os.path.isdir(os.path.join(NEG_ROOT, d))]

for folder in tqdm(user_folders, desc="Sampling negative tweets"):
    tweets = extract_user_tweets(folder)
    
    for row in tweets:
        count += 1
        if len(sample) < N_POSITIVE:
            sample.append(row)
        else:
            # Replace random element with decreasing probability
            r = random.randint(0, count - 1)
            if r < N_POSITIVE:
                sample[r] = row

print(f"Total negative tweets processed: {count}")
print(f"Final sample size: {len(sample)}")

# Save to Parquet
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)  # ensure folder exists
df_neg = pd.DataFrame(sample)
df_neg.to_parquet(OUTPUT_FILE, engine="pyarrow", index=False)
print(f"Saved balanced English negative sample to {OUTPUT_FILE}")