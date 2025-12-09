import os
import json
import pandas as pd
from tqdm import tqdm
from dateutil import parser
import pytz  # for UTC handling

# Path to your depression JSON data
DATA_ROOT = "data/depression/media/data_dump/udit/submission/dataset/depression/cleaned_data_depression"
TIME_WINDOW_DAYS = 90  # ±90 days around anchor tweet

all_rows = []

def make_aware(dt_str):
    """Convert a datetime string to an offset-aware datetime in UTC."""
    dt = parser.isoparse(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.UTC)
    else:
        dt = dt.astimezone(pytz.UTC)
    return dt

# Iterate over all user folders
for root, dirs, files in tqdm(os.walk(DATA_ROOT), desc="Processing folders"):
    if "user.json" in files and "anchor_tweet.json" in files and "tweets.json" in files:
        try:
            # Load user info
            with open(os.path.join(root, "user.json")) as f:
                user = json.load(f)
            user_id = user.get("id")

            # Load anchor tweet
            with open(os.path.join(root, "anchor_tweet.json")) as f:
                anchor = json.load(f)
            anchor_text = anchor.get("anchor_tweet")
            anchor_date_str = anchor.get("anchor_tweet_date")

            # Skip users without anchor
            if not anchor_text or not anchor_date_str:
                continue

            anchor_date = make_aware(anchor_date_str)

            # Add anchor tweet
            all_rows.append({
                "user_id": user_id,
                "tweet_text": anchor_text,
                "tweet_date": anchor_date_str,
                "is_anchor": True,
                "label": "depressed"
            })

            # Load user tweets
            with open(os.path.join(root, "tweets.json")) as f:
                tweets = json.load(f)

            # Flatten tweets.json
            for date_str, tweets_list in tweets.items():
                for tweet in tweets_list:
                    tweet_text = tweet.get("text")
                    tweet_date = tweet.get("timestamp_tweet")
                    if not tweet_text or not tweet_date:
                        continue

                    tweet_dt = make_aware(tweet_date)
                    delta_days = abs((tweet_dt - anchor_date).days)

                    if delta_days <= TIME_WINDOW_DAYS:
                        all_rows.append({
                            "user_id": user_id,
                            "tweet_text": tweet_text,
                            "tweet_date": tweet_date,
                            "is_anchor": False,
                            "label": "depressed"
                        })

        except Exception as e:
            print(f"Error in folder {root}: {e}")
            continue

# Convert to DataFrame and save as Parquet
df_tweets = pd.DataFrame(all_rows)
output_path = "data/tweets.parquet"
df_tweets.to_parquet(output_path, engine='pyarrow', index=False)
print(f"Saved {df_tweets.shape[0]} tweets to {output_path}")