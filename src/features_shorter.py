# This is ok, but we modified it - this one has a lot of empath categories. 

import re
import emoji
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
from empath import Empath
import textstat
#this one is a troublemaker... 
#import spacy
from collections import Counter

tqdm.pandas()
lexicon = Empath()
#nlp = spacy.load("en_core_web_sm")

FIRST_PERSON = re.compile(r'\b(i|me|my|mine|myself)\b', flags=re.IGNORECASE)
NEGATIONS = re.compile(r'\b(not|no|never|none|nobody|nothing|neither)\b', flags=re.IGNORECASE)

def count_uppercase_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    upper_words = sum(1 for w in words if w.isupper())
    return upper_words / len(words)

def count_elongation(text: str) -> int:
    return len(re.findall(r'(.)\1{2,}', text))

def count_ellipsis(text: str) -> int:
    return text.count("...")

def count_punctuations(text: str) -> dict:
    return {"count_exclamation": text.count("!"),
            "count_question": text.count("?")}

def count_emojis(text: str) -> dict:
    emojis = [c for c in text if c in emoji.EMOJI_DATA]
    return {"emoji_count": len(emojis), "emoji_unique_count": len(set(emojis))}

def get_sentiment(text: str) -> dict:
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity}

def get_empath_features(text: str) -> dict:
    categories = ["sadness", "fear", "anger", "negative_emotion", "positive_emotion",
                  "social", "family", "friendship", "swearing", "health", "money",
                  "work", "leisure", "time", "past", "future", "cognitive_mechanisms",
                  "perceptual", "motion", "body", "sexual", "religion", "affection",
                  "achievement", "reward", "dominance", "power", "anxious", "tentative", "certainty"]
    return lexicon.analyze(text, categories=categories)

def count_first_person(text: str) -> int:
    return len(FIRST_PERSON.findall(text))

def count_negations(text: str) -> int:
    return len(NEGATIONS.findall(text))

def count_repeated_words(text: str) -> int:
    words = text.split()
    counts = Counter(words)
    return counts.most_common(1)[0][1] if counts else 0

# spaCy incompatible with python version, commented out for now
#def count_past_future_verbs(text: str) -> dict:
#    doc = nlp(text)
#    past = sum(1 for tok in doc if tok.tag_ in ["VBD", "VBN"])
#    future = sum(1 for tok in doc if tok.tag_ == "MD" and tok.text.lower() in ["will", "shall"])
#    return {"past_verbs": past, "future_verbs": future}

def get_readability(text: str) -> dict:
    try:
        flesch = textstat.flesch_reading_ease(text)
        grade = textstat.flesch_kincaid_grade(text)
    except:
        flesch = 0
        grade = 0
    return {"flesch_reading_ease": flesch, "flesch_kincaid_grade": grade}

def extract_features(df: pd.DataFrame, text_column="clean_text") -> pd.DataFrame:
    df = df.copy()
    df["char_count"] = df[text_column].str.len()
    df["word_count"] = df[text_column].str.split().str.len()
    df["avg_word_len"] = df.apply(lambda row: (len(row[text_column])/row["word_count"])
                                  if row["word_count"] > 0 else 0, axis=1)
    df["unique_word_ratio"] = df.apply(lambda row: (len(set(row[text_column].split()))/row["word_count"])
                                       if row["word_count"] > 0 else 0, axis=1)
    df["uppercase_ratio"] = df[text_column].progress_apply(count_uppercase_ratio)
    df["elongation_count"] = df[text_column].progress_apply(count_elongation)
    df["ellipsis_count"] = df[text_column].progress_apply(count_ellipsis)
    punct_df = df[text_column].progress_apply(count_punctuations).apply(pd.Series)
    df = pd.concat([df, punct_df], axis=1)
    emoji_df = df[text_column].progress_apply(count_emojis).apply(pd.Series)
    df = pd.concat([df, emoji_df], axis=1)
    sentiment_df = df[text_column].progress_apply(get_sentiment).apply(pd.Series)
    df = pd.concat([df, sentiment_df], axis=1)
    empath_df = df[text_column].progress_apply(get_empath_features).apply(pd.Series)
    df = pd.concat([df, empath_df], axis=1)
    df["first_person_count"] = df[text_column].progress_apply(count_first_person)
    df["negation_count"] = df[text_column].progress_apply(count_negations)
    df["max_word_repeat"] = df[text_column].progress_apply(count_repeated_words)
    # spaCy incompatible with python version, commented out for now
    #verb_df = df[text_column].progress_apply(count_past_future_verbs).apply(pd.Series)
    #df = pd.concat([df, verb_df], axis=1)
    readability_df = df[text_column].progress_apply(get_readability).apply(pd.Series)
    df = pd.concat([df, readability_df], axis=1)
    return df

if __name__ == "__main__":
    df = pd.read_parquet("data/shorty_clean.parquet")
    df_feat = extract_features(df)
    df_feat.to_parquet("data/shorty_features.parquet", index=False)
    print(f"Features extracted and saved to data/shorty_features.parquet. Total rows: {df_feat.shape[0]}")