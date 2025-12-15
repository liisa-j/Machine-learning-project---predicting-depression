## Preprocessing: 7 ## 
## This code combines tweets by user
## Inputs shorty_clean.parquet and outputs twitter_combineduser.parquet


import pandas as pd

df = pd.read_parquet("data/shorty_clean.parquet")

user_df = (
    df
    .groupby("user_id", as_index=False)
    .agg({
        "clean_text": " ".join,
        "language": "first",
        "label_encoded": "first"
    })
)

output_path = "data/twitter_combineduser.parquet"
user_df.to_parquet(output_path, index=False)

print(f"Saved combined user-level dataset to {output_path}")