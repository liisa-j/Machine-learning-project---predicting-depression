import re
import emoji
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

tqdm.pandas()

def preprocess_tweet(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "<URL>", text)
    text = re.sub(r"@\w+", "@", text)
    text = re.sub(r"\brt\b", "", text, flags=re.IGNORECASE)
    text = emoji.demojize(text)
    text = " ".join(text.split())
    return text

def preprocess_dataframe(df: pd.DataFrame, text_column: str = "tweet_text") -> pd.DataFrame:
    df = df.copy()
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available columns: {df.columns.tolist()}")
    if "label" in df.columns:
        le = LabelEncoder()
        df["label_encoded"] = le.fit_transform(df["label"])
    df["clean_text"] = df[text_column].astype(str).progress_apply(preprocess_tweet)
    df = df.drop(columns=[c for c in ["is_anchor", text_column, "label"] if c in df.columns])
    return df

if __name__ == "__main__":
    df = pd.read_parquet("data/shorty.parquet")
    df_clean = preprocess_dataframe(df)
    df_clean.to_parquet("data/shorty_clean.parquet", index=False)
    print("Ready! Dataframe saved as - data/shorty_clean.parquet")