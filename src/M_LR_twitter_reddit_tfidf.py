from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Load Reddit and Twitter features
df_reddit = pd.read_parquet("data/reddit_features.parquet")
df_twitter = pd.read_parquet("data/shorty_features.parquet")

# Combine numeric features
NUMERIC_FEATURES = [col for col in df_reddit.columns if col not in ["author", "clean_text", "label_encoded"]]

# TF-IDF on text
tfidf = TfidfVectorizer(max_features=5000)  # you can adjust size
X_twitter_text = tfidf.fit_transform(df_twitter["clean_text"])
X_reddit_text = tfidf.transform(df_reddit["clean_text"])

# Standardize numeric features
scaler = StandardScaler()
X_twitter_num = scaler.fit_transform(df_twitter[NUMERIC_FEATURES])
X_reddit_num = scaler.transform(df_reddit[NUMERIC_FEATURES])

# Combine numeric + TF-IDF
from scipy.sparse import hstack
X_train = hstack([X_twitter_num, X_twitter_text])
y_train = df_twitter["label_encoded"]
X_test = hstack([X_reddit_num, X_reddit_text])
y_test = df_reddit["label_encoded"]

# Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))