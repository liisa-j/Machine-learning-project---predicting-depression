import pandas as pd

input_file = "../data/combined_tweets.parquet"
output_file = "../data/shorty.parquet"
sample_size = 1_000_000  # target number of rows

print("Loading dataset...")
df = pd.read_parquet(input_file, engine="pyarrow")

# sample 1 million rows
df_sampled = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

# save to parquet
print(f"Saving sampled dataset to {output_file}...")
df_sampled.to_parquet(output_file, engine="pyarrow", index=False)
print(f"Done! Total rows: {df_sampled.shape[0]}")