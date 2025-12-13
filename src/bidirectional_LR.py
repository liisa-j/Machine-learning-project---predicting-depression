
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack


twitter_file = "data/shorty_features.parquet"
reddit_file = "data/reddit_features.parquet"


df_twitter = pd.read_parquet(twitter_file)
df_reddit = pd.read_parquet(reddit_file)


text_col = "clean_text"
label_col = "label_encoded"


numeric_features = [col for col in df_twitter.columns 
                    if col not in [text_col, label_col, "user_id", "author", "tweet_date", "language"]]


# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 1)) 

X_twitter_tfidf = tfidf.fit_transform(df_twitter[text_col])
X_reddit_tfidf = tfidf.transform(df_reddit[text_col])


scaler = StandardScaler(with_mean=False)
X_twitter_num = scaler.fit_transform(df_twitter[numeric_features])
X_reddit_num = scaler.transform(df_reddit[numeric_features])


X_twitter = hstack([X_twitter_tfidf, X_twitter_num])
X_reddit = hstack([X_reddit_tfidf, X_reddit_num])


y_twitter = df_twitter[label_col]
y_reddit = df_reddit[label_col]


def train_test_LR(X_train, y_train, X_test, y_test, description=""):
    print(f"\nTraining on {description}...")
    clf = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced', n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))


train_test_LR(X_twitter, y_twitter, X_reddit, y_reddit, "Twitter, Testing on Reddit")


train_test_LR(X_reddit, y_reddit, X_twitter, y_twitter, "Reddit, Testing on Twitter")