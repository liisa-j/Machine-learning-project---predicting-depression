import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# File paths
twitter_feat_file = "data/shorty_features.parquet"
reddit_feat_file = "data/reddit_features.parquet"


# Load datasets
df_twitter = pd.read_parquet(twitter_feat_file)
df_reddit = pd.read_parquet(reddit_feat_file)


# Select numeric features only
def get_feature_columns(df):
    return df.select_dtypes(include=["float64", "int64"]).columns.drop("label_encoded").tolist()

FEATURE_COLS = list(set(get_feature_columns(df_twitter)) & set(get_feature_columns(df_reddit)))


# Train/test split
X_train = df_twitter[FEATURE_COLS]
y_train = df_twitter["label_encoded"]

X_test = df_reddit[FEATURE_COLS]
y_test = df_reddit["label_encoded"]


# Train Logistic Regression
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)


# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))