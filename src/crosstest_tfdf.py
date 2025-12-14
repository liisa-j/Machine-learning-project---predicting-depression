import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# File paths for the Twitter and Reddit datasets
twitter_file = "data/shorty_features.parquet"
reddit_file = "data/reddit_features.parquet"

# Load the datasets
df_twitter = pd.read_parquet(twitter_file)
df_reddit = pd.read_parquet(reddit_file)

# Column names for text and label
text_col = "clean_text"
label_col = "label_encoded"

# TF-IDF vectorization (Only text features)
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 1))

# Transform the text data into TF-IDF matrices
X_twitter_tfidf = tfidf.fit_transform(df_twitter[text_col])
X_reddit_tfidf = tfidf.transform(df_reddit[text_col])

# Target labels for classification
y_twitter = df_twitter[label_col]
y_reddit = df_reddit[label_col]

# Function to train and evaluate a Logistic Regression model
def train_test_LR(X_train, y_train, X_test, y_test, description=""):
    print(f"\nTraining on {description}...")
    clf = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced', n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

# Train on Twitter data and test on Reddit data
train_test_LR(X_twitter_tfidf, y_twitter, X_reddit_tfidf, y_reddit, "Twitter, Testing on Reddit")

# Train on Reddit data and test on Twitter data
train_test_LR(X_reddit_tfidf, y_reddit, X_twitter_tfidf, y_twitter, "Reddit, Testing on Twitter")