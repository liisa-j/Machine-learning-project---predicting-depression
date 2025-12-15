## Preprocessing: 8 ## Optional - wasn't used
## This script truncates the longest texts (at 95t percentile)
## Inputs shorty_clean outputs twitter_combineduser_trunc.parquet


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

# Truncate clean_text at 95th percentile (by word count)
user_df['word_count'] = user_df['clean_text'].str.split().str.len()
MAX_WORDS = int(user_df['word_count'].quantile(0.95))
print(f"Truncating texts to max {MAX_WORDS} words (95th percentile)")

user_df['clean_text'] = user_df['clean_text'].apply(
    lambda x: ' '.join(x.split()[:MAX_WORDS])
)

user_df.drop(columns=['word_count'], inplace=True)

output_path = "data/twitter_combineduser_trunc.parquet"
user_df.to_parquet(output_path, index=False)

print(f"Saved combined user-level dataset to {output_path}")
