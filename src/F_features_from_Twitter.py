## Feature extraction: Twitter ##
## This is the script we are using to extract features from Twitter data. 
## Has less empath categories + some added features. 
### Takes in shorty_clean.parquet and outputs shorty_features.parquet


import re
import emoji
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
from empath import Empath
import textstat
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

tqdm.pandas()
lexicon = Empath()
analyzer = SentimentIntensityAnalyzer()
nltk.download('punkt')


FIRST_PERSON = re.compile(r'\b(i|me|my|mine|myself)\b', flags=re.IGNORECASE)
NEGATIONS = re.compile(r'\b(not|no|never|none|nobody|nothing|neither)\b', flags=re.IGNORECASE)
REPEATED_EXCL = re.compile(r'!{2,}')
REPEATED_QMARK = re.compile(r'\?{2,}')


FUNCTION_WORDS = ["you", "we", "they", "he", "she", "it", "him", "her", "us", "them", "this", "that", "these", "those"]

# feature functions
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
            "count_question": text.count("?"),
            "repeated_exclamation": len(REPEATED_EXCL.findall(text)),
            "repeated_question": len(REPEATED_QMARK.findall(text))}


EMOJI_DEMOTIF = re.compile(r':([a-zA-Z_]+):')

def count_emojis(text: str) -> dict:
    emojis = EMOJI_DEMOTIF.findall(text)
    return {"emoji_count": len(emojis), "emoji_unique_count": len(set(emojis))}


def get_sentiment(text: str) -> dict:
    blob = TextBlob(text)
    vader_scores = analyzer.polarity_scores(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "vader_neg": vader_scores['neg'],
        "vader_neu": vader_scores['neu'],
        "vader_pos": vader_scores['pos'],
        "vader_compound": vader_scores['compound']
    }

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

def count_function_words(text: str) -> int:
    words = text.lower().split()
    return sum(words.count(f) for f in FUNCTION_WORDS)

def count_sentences(text: str) -> int:
    from nltk.tokenize import sent_tokenize
    return len(sent_tokenize(text))

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
    df["sentence_count"] = df[text_column].progress_apply(count_sentences)
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
    df["function_word_count"] = df[text_column].progress_apply(count_function_words)
    readability_df = df[text_column].progress_apply(get_readability).apply(pd.Series)
    df = pd.concat([df, readability_df], axis=1)
    return df

if __name__ == "__main__":
    df = pd.read_parquet("data/shorty_clean.parquet")
    df_feat = extract_features(df)
    df_feat.to_parquet("data/shorty_features.parquet", index=False)
    print(f"Features extracted and saved to data/shorty_features.parquet. Total rows: {df_feat.shape[0]}")