## This script runs Random Forest on reddit feature extracted dataset
## Takes in reddit_features.parquet

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_parquet("data/reddit_features.parquet")

y = df["label_encoded"]
X = df.drop(columns=["label_encoded", "clean_text"]) 

non_numeric = X.select_dtypes(exclude='number').columns.tolist()
if non_numeric:
    print("Warning: non-numeric columns detected, they will be dropped:", non_numeric)
    X = X.drop(columns=non_numeric)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=500, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

feature_importances = clf.feature_importances_
feature_names = X.columns.tolist()

top_n = 20
top_indices = np.argsort(feature_importances)[-top_n:]
top_features = [feature_names[i] for i in top_indices]
top_importances = feature_importances[top_indices]

plt.figure(figsize=(10,6))
plt.barh(top_features, top_importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.title(f"Top {top_n} Most Important Numeric Features for Random Forest")
plt.tight_layout()
plt.show()

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

top_indices = np.argsort(feature_importances)[-top_n:] 
top_features = [feature_names[i] for i in top_indices]
top_importances = feature_importances[top_indices]

plt.figure(figsize=(10, 6))
plt.barh(top_features, top_importances, color='lightgreen')
plt.xlabel('Feature Importance')
plt.title(f'Top {top_n} Feature Importances')
plt.tight_layout()
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_test, cmap='coolwarm', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D PCA Projection of Test Set')
plt.colorbar(label='Depressed Label')
plt.show()
