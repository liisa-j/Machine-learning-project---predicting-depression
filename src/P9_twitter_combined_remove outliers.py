### Preprocessing: 9 ### Optional - wasn't used
### This code removes outliers above 95th percentile (by word count)
### Inputs twitter_combineduser.parquet outputs twitter_combineduser_no_outliers.parquet


import pandas as pd


df = pd.read_parquet("data/twitter_combineduser.parquet")

df['word_count'] = df['clean_text'].str.split().str.len()

def remove_outliers(group):
    threshold = group['word_count'].quantile(0.95)
    return group[group['word_count'] <= threshold]

df_clean = df.groupby('label_encoded', group_keys=False).apply(remove_outliers)
df_clean.drop(columns=['word_count'], inplace=True)

output_path = "data/twitter_combineduser_no_outliers.parquet"
df_clean.to_parquet(output_path, index=False)

print(f"Removed outliers and saved dataset to {output_path}")
