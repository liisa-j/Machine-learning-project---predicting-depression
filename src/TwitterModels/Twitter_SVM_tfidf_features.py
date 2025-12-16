### This script runs SVM on Twitter tweet level data using both tf-idf and calculated features
# Inputs shorty_feature.parquet
# Be informed - this is very slow... 

import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix


print("Loading data...")
df = pd.read_parquet("data/shorty_features.parquet")

text_col = "clean_text"

drop_cols = ["clean_text", "label", "label_encoded", "user_id", "tweet_date", "language"]
numeric_cols = [c for c in df.columns if c not in drop_cols]

print(f"Using {len(numeric_cols)} numeric features")


X_text = df[text_col].fillna("")
X_num = df[numeric_cols].fillna(0)
y = df["label_encoded"].astype(int)


print("Vectorizing text... (TF-IDF 1-2 grams)")
tfidf = TfidfVectorizer(
    max_features=200000,  
    ngram_range=(1, 2),   
    min_df=3,             
)
X_text_tfidf = tfidf.fit_transform(X_text)


print("Scaling numeric features...")
scaler = StandardScaler(with_mean=False) 
X_num_scaled = scaler.fit_transform(X_num)


print("Combining features...")
X_full = hstack([X_text_tfidf, X_num_scaled])


print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)


print("Training Linear SVM...")
start = time.time()
clf = LinearSVC(max_iter=10000)
clf.fit(X_train, y_train)
end = time.time()


print("Evaluating...")
y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nTotal runtime: {round((end - start) / 60, 2)} minutes")