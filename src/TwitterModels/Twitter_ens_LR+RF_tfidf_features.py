## Ensemble: Numeric + TF-IDF features on Twitter data
## Inputs: shorty_features.parquet

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from scipy.sparse import hstack
import time

start_time = time.time()

# Load data
df = pd.read_parquet("data/shorty_features.parquet")

# Split features
X_numeric = df.drop(columns=['user_id','tweet_date','language','clean_text','label_encoded'])
y = df['label_encoded']
text = df['clean_text']

# Train/test split
X_train_num, X_test_num, y_train, y_test, text_train, text_test = train_test_split(
    X_numeric, y, text, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num.astype(np.float32))
X_test_num_scaled = scaler.transform(X_test_num.astype(np.float32))

# TF-IDF vectorizer (word 1-3 ngrams)
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,3))
X_train_text = tfidf.fit_transform(text_train)
X_test_text = tfidf.transform(text_test)

# Combine features for final stacking
X_train_final = hstack([X_train_num_scaled, X_train_text])
X_test_final = hstack([X_test_num_scaled, X_test_text])

# Define base models
numeric_model = RandomForestClassifier(
    n_estimators=200, max_depth=40, min_samples_split=10, min_samples_leaf=5,
    max_features='sqrt', n_jobs=-1, random_state=42, verbose=0
)

text_model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)

# Stacking ensemble
estimators = [
    ('rf_numeric', numeric_model),
    ('lr_text', text_model)
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    n_jobs=-1,
    passthrough=True
)

# Train ensemble
print("Training ensemble...")
stack_model.fit(X_train_final, y_train)

# Predict
y_pred = stack_model.predict(X_test_final)
y_proba = stack_model.predict_proba(X_test_final)[:,1]

# Metrics
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test, y_proba):.4f}")

end_time = time.time()
print(f"\nTotal runtime: {(end_time - start_time)/60:.2f} minutes")
