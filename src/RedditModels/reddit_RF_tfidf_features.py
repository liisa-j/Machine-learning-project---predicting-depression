## This script runs Random Forest on reddit dataset using Tf-idf and extracted features
## input is reddit_features.parquet

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import hstack, csr_matrix
from wordcloud import WordCloud

df = pd.read_parquet("data/reddit_features.parquet")

y = df['label_encoded']
X_num = df.drop(columns=['label_encoded', 'clean_text'], errors='ignore')
X_text = df['clean_text'].astype(str)

non_numeric = X_num.select_dtypes(exclude='number').columns.tolist()
if non_numeric:
    X_num = X_num.drop(columns=non_numeric)

tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,3), min_df=5, max_features=100000)
X_tfidf = tfidf.fit_transform(X_text)

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)
X_num_sparse = csr_matrix(X_num_scaled)  
X_combined = hstack([X_num_sparse, X_tfidf])

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

if len(np.unique(y)) == 2:
    y_score = clf.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

feature_importances = clf.feature_importances_
feature_names_num = X_num.columns.tolist()
coef_num = feature_importances[:X_num.shape[1]]

top_n = 20
top_indices = np.argsort(coef_num)[-top_n:]
top_features = [feature_names_num[i] for i in top_indices]
top_coefs = coef_num[top_indices]

plt.figure(figsize=(10,6))
colors = ['green' if c>0 else 'red' for c in top_coefs]
plt.barh(top_features, top_coefs, color=colors)
plt.xlabel("Importance")
plt.title(f"Top {top_n} Numeric Feature Importances")
plt.tight_layout()
plt.show()



