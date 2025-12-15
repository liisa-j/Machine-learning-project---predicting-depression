## This is an attempt to use DepecheMood to extract features from 
# Reddit data in addition to other sentiment lexicons - wasn't used

import re
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
from empath import Empath
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
from collections import Counter
import nltk
import textacy.resources

tqdm.pandas()
nltk.download("punkt")


lexicon = Empath()
vader = SentimentIntensityAnalyzer()
depechemood = textacy.resources.DepecheMood(lang="en", word_rep="lemmapos")
depechemood.download()  


FIRST_PERSON = re.compile(r'\b(i|me|my|mine|myself)\b', flags=re.IGNORECASE)
NEGATIONS = re.compile(r'\b(not|no|never|none|nobody|nothing|neither)\b', flags=re.IGNORECASE)
REPEATED_EXCL = re.compile(r'!{2,}')
REPEATED_QMARK = re.compile(r'\?{2,}')

FUNCTION_WORDS = ["you", "we", "they", "he", "she", "it", "him", "her", "us", "them",
                  "this", "that", "these", "those"]

ABSOLUTIST_WORDS = ["always", "never", "entire", "totally"]



def count_uppercase_ratio(text):
    words = text.split()
    return sum(w.isupper() for w in words)/len(words) if words else 0

def count_elongation(text):
    return len(re.findall(r'(.)\1{2,}', text))

def count_ellipsis(text):
    return text.count("...")

def count_punctuations(text):
    return {
        "count_exclamation": text.count("!"),
        "count_question": text.count("?"),
        "repeated_exclamation": len(REPEATED_EXCL.findall(text)),
        "repeated_question": len(REPEATED_QMARK.findall(text))
    }

def count_emojis(text):
    return {"emoji_count": len(re.findall(r':[a-zA-Z_]+:', text))}

def get_sentiment(text):
    blob = TextBlob(text)
    vader_scores = vader.polarity_scores(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "vader_neg": vader_scores['neg'],
        "vader_neu": vader_scores['neu'],
        "vader_pos": vader_scores['pos'],
        "vader_compound": vader_scores['compound']
    }

def get_empath_features(text):
    categories = ["sadness", "fear", "anger", "negative_emotion", "positive_emotion",
                  "social", "swearing", "time", "past", "future", "cognitive_mechanisms",
                  "perceptual", "affection", "achievement", "reward", "dominance",
                  "power", "anxious", "tentative", "certainty"]
    return lexicon.analyze(text, categories=categories)

def get_depechemood_features(text, lexicon=depechemood):
    words = re.findall(r'\b\w+\b', text.lower())
    scores = {emotion: 0.0 for emotion in lexicon.emotions}
    count = 0
    for w in words:
        token_key = w + "#n"  # naive noun POS; for improved accuracy, can POS-tag
        if token_key in lexicon._dm:
            for emotion in lexicon.emotions:
                scores[emotion] += lexicon._dm[token_key].get(emotion, 0)
            count += 1
    if count > 0:
        for emotion in scores:
            scores[emotion] /= count
    return scores

def absolutist_ratio(text):
    words = text.split()
    if not words:
        return 0
    count = sum(1 for w in words if w.lower() in ABSOLUTIST_WORDS)
    return count / len(words)

def count_first_person(text):
    return len(FIRST_PERSON.findall(text))

def count_negations(text):
    return len(NEGATIONS.findall(text))

def count_repeated_words(text):
    words = text.split()
    counts = Counter(words)
    return counts.most_common(1)[0][1] if counts else 0

def count_function_words(text):
    words = text.lower().split()
    return sum(words.count(f) for f in FUNCTION_WORDS)

def count_sentences(text):
    from nltk.tokenize import sent_tokenize
    return len(sent_tokenize(text))

def get_readability(text):
    try:
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text)
        }
    except:
        return {"flesch_reading_ease": 0, "flesch_kincaid_grade": 0}


def extract_features(df, text_column="clean_text"):
    df = df.copy()
    
   
    df["absolutist_ratio"] = df[text_column].progress_apply(absolutist_ratio)
    df["char_count"] = df[text_column].str.len()
    df["word_count"] = df[text_column].str.split().str.len()
    df["sentence_count"] = df[text_column].progress_apply(count_sentences)
    df["avg_word_len"] = df["char_count"] / df["word_count"].replace(0, 1)
    df["unique_word_ratio"] = df.apply(lambda row: len(set(row[text_column].split())) / row["word_count"]
                                       if row["word_count"] > 0 else 0, axis=1)

    
    df["uppercase_ratio"] = df[text_column].progress_apply(count_uppercase_ratio)
    df["elongation_count"] = df[text_column].progress_apply(count_elongation)
    df["ellipsis_count"] = df[text_column].progress_apply(count_ellipsis)
    punct_df = df[text_column].progress_apply(count_punctuations).apply(pd.Series)
    df = pd.concat([df, punct_df], axis=1)

   
    emoji_df = df[text_column].progress_apply(count_emojis).apply(pd.Series)
    df = pd.concat([df, emoji_df], axis=1)
    df["emoji_ratio"] = df["emoji_count"] / df["word_count"].replace(0, 1)

    
    sentiment_df = df[text_column].progress_apply(get_sentiment).apply(pd.Series)
    df = pd.concat([df, sentiment_df], axis=1)

    
    empath_df = df[text_column].progress_apply(get_empath_features).apply(pd.Series)
    for col in empath_df.columns:
        empath_df[col] = empath_df[col] / df["word_count"].replace(0,1)
    df = pd.concat([df, empath_df], axis=1)

   
    dm_df = df[text_column].progress_apply(get_depechemood_features).apply(pd.Series)
    df = pd.concat([df, dm_df], axis=1)

  
    df["first_person_ratio"] = df[text_column].progress_apply(count_first_person) / df["word_count"].replace(0,1)
    df["negation_ratio"] = df[text_column].progress_apply(count_negations) / df["word_count"].replace(0,1)
    df["max_word_repeat"] = df[text_column].progress_apply(count_repeated_words)
    df["function_word_ratio"] = df[text_column].progress_apply(count_function_words) / df["word_count"].replace(0,1)

    
    readability_df = df[text_column].progress_apply(get_readability).apply(pd.Series)
    df = pd.concat([df, readability_df], axis=1)

    return df



if __name__ == "__main__":
    df = pd.read_parquet("data/reddit_text_classification_dataset.parquet")
    df_feat = extract_features(df)
    df_feat.to_parquet("data/reddit_featurized.parquet", index=False)
    print(f"Features extracted and saved to data/reddit_featurized.parquet. Total rows: {df_feat.shape[0]}")
