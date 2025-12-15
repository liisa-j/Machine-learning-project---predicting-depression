##This script runs SVM on reddit dataset using tfidf vectorization only
## Inputs reddit_text_classification_dataset.parquet

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
df = pd.read_parquet("data/reddit_text_classification_dataset.parquet")

# Prepare text and labels
X = df["clean_text"].astype(str)
y = df["label_encoded"].astype(int)

print("Vectorizing text...")
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),        
    analyzer='word',           
    min_df=5,
    max_features=300000
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

## Feature Importance
print("Plotting feature importance...")
coef = clf.coef_.flatten() 

feature_names = tfidf.get_feature_names_out()
top_indices = np.argsort(np.abs(coef))[-20:]  
top_terms = [feature_names[i] for i in top_indices]
top_coefs = coef[top_indices]

plt.figure(figsize=(10, 6))
plt.barh(top_terms, top_coefs)
plt.xlabel("Coefficient Value")
plt.title("Top 20 Important Features (Terms) for Linear SVM")
plt.show()

#Error Analysis
# misclassified samples
misclassified_idx = np.where(pred != y_test)[0]
misclassified_samples = X.iloc[misclassified_idx] 
misclassified_true_labels = y_test.iloc[misclassified_idx]
misclassified_pred_labels = pred[misclassified_idx]


for i in range(5): 
    print(f"True label: {misclassified_true_labels.iloc[i]}, Predicted label: {misclassified_pred_labels[i]}")
    print(f"Text: {misclassified_samples.iloc[i]}")  
    print("-" * 80)


print(f"Total runtime: {round((time.time() - start)/60, 2)} minutes")