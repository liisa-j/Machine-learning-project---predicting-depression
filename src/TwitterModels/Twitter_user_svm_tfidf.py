## This script runs linear SVM on Twitter user level data  
## Inputs features_combinedtwitter.parquet

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


start = time.time()

print("Loading data...")
df = pd.read_parquet("data/features_combinedtwitter.parquet")

X = df["clean_text"].astype(str)
y = df["label_encoded"].astype(int)

print("Vectorizing text...")
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),        
    analyzer='word',           
    min_df=5,
    max_features=30000
)

X_tfidf = tfidf.fit_transform(X)

print("Splitting...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

print("Training Linear SVM...")
clf = LinearSVC()
clf.fit(X_train, y_train)

print("Evaluating...")
pred = clf.predict(X_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
print(f"Total runtime: {round((time.time() - start)/60,2)} minutes")


# Get coefficients
coef = clf.coef_.flatten()
feature_names = tfidf.get_feature_names_out()

# Top 20 features by absolute value
top_n = 20
top_indices = np.argsort(np.abs(coef))[-top_n:]
top_features = feature_names[top_indices]
top_coefs = coef[top_indices]

# Plot
plt.figure(figsize=(10,6))
colors = ['red' if c>0 else 'green' for c in top_coefs]
plt.barh(top_features, top_coefs, color=colors)
plt.xlabel("Coefficient Value")
plt.title(f"Top {top_n} TF-IDF Features for Linear SVM")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


