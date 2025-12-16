#This script uses user level Tweets and combines both tfidf and numeric features
# And runs RF
# input is features_combinedtwitter.parquet 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


df = pd.read_parquet("data/features_combinedtwitter.parquet")
print("Dataset shape:", df.shape)
print("Class distribution:\n", df['label_encoded'].value_counts())


numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
X_numeric = df[numeric_cols].drop(columns=['label_encoded'], errors='ignore')
y = df['label_encoded']


scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)


text_col = 'clean_text'
X_text = df[text_col].fillna("").astype(str)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=5)
X_text_tfidf = tfidf.fit_transform(X_text)


X_combined = hstack([X_numeric_scaled, X_text_tfidf])


X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)


model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
print("Training Random Forest...")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


importances = model.feature_importances_
feature_names = list(X_numeric.columns) + list(tfidf.get_feature_names_out())


indices = np.argsort(importances)[::-1]
top_n = 20
top_features = [feature_names[i] for i in indices[:top_n]]
top_importances = importances[indices[:top_n]]


colors = ['green' if i < X_numeric.shape[1] else 'blue' for i in indices[:top_n]]

plt.figure(figsize=(10,8))
plt.barh(top_features[::-1], top_importances[::-1], color=colors[::-1])
plt.xlabel("Feature Importance")
plt.title("Top 20 Feature Importances (Green=Numeric, Blue=TF-IDF)")
plt.tight_layout()
plt.show()