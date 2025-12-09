import os
import json
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# Extract tweets in batches (because the data is huge)

NEG_ROOT = "data/neg_data/media/data_dump/udit/submission/dataset/neg/cleaned_data_neg"
OUTPUT_DIR = "data/neg_parquet_batches"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BATCH_SIZE = 500

def extract_user_tweets(user_folder):
    tweets_path = os.path.join(user_folder, "tweets.json")
    if not os.path.exists(tweets_path):
        return pd.DataFrame()

    try:
        with open(tweets_path, "r") as f:
            tweets_dict = json.load(f)
    except:
        return pd.DataFrame()

    rows = []
    for date, tweets_list in tweets_dict.items():
        for t in tweets_list:
            rows.append({
                "tweet_id": t.get("tweet_id"),
                "text": t.get("text"),
                "timestamp": t.get("timestamp_tweet"),
                "disorder_flag": t.get("disorder_flag", False),
                "user_id": os.path.basename(user_folder)
            })
    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame()

user_folders = [os.path.join(NEG_ROOT, d) for d in os.listdir(NEG_ROOT) if os.path.isdir(os.path.join(NEG_ROOT, d))]

for i in tqdm(range(0, len(user_folders), BATCH_SIZE), desc="Processing negative dataset in batches"):
    batch_folders = user_folders[i:i+BATCH_SIZE]
    batch_dfs = [extract_user_tweets(f) for f in batch_folders]
    batch_df = pd.concat(batch_dfs, ignore_index=True)

    if not batch_df.empty:
        batch_file = os.path.join(OUTPUT_DIR, f"neg_batch_{i//BATCH_SIZE + 1}.parquet")
        batch_df.to_parquet(batch_file, engine='pyarrow', index=False)

    del batch_df

# Combine all batch Parquet files into one file (neg_combined.parquet)

all_files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith(".parquet")]
output_file = "data/neg_combined.parquet"

first_file = True
for f in tqdm(all_files, desc="Combining batch Parquet files"):
    df = pd.read_parquet(f)
    table = pa.Table.from_pandas(df)

    if first_file:
        writer = pq.ParquetWriter(output_file, table.schema)
        first_file = False

    writer.write_table(table)

writer.close()
print("All negative tweets combined into:", output_file)