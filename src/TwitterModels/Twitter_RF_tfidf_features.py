## This code runs Random Forest on Twitter dataset and uses Tf-Idf and extracted features
## Inputs shorty_features.parquet
## Be informed - Super slow.... 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time


df = pd.read_parquet("data/shorty_features.parquet")


X_numeric = df.drop(columns=['user_id','tweet_date','language','clean_text','label_encoded'])
y = df['label_encoded']
text = df['clean_text']


X_train_num, X_test_num, y_train, y_test, text_train, text_test = train_test_split(
    X_numeric, y, text, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# TF-IDF vectorizer for n-grams
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_text = tfidf.fit_transform(text_train)
X_test_text = tfidf.transform(text_test)


X_train_final = hstack([X_train_num_scaled, X_train_text])
X_test_final = hstack([X_test_num_scaled, X_test_text])


clf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
print("Training Random Forest with TF-IDF n-grams + numeric features...")
start = time.time()
clf.fit(X_train_final, y_train)
end = time.time()


y_pred = clf.predict(X_test_final)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Total runtime: {round((end-start)/60, 2)} minutes")