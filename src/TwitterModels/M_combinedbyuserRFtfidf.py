import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


df = pd.read_parquet("data/features_combinedtwitter.parquet")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
X_numeric = df[numeric_cols].drop(columns=['label_encoded'], errors='ignore')


tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_tfidf = tfidf.fit_transform(df['clean_text'])


X = hstack([X_numeric, X_tfidf])
y = df['label_encoded']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


importances = model.feature_importances_[:X_numeric.shape[1]]  
indices = np.argsort(importances)[::-1]  

top_n = 20
features = [X_numeric.columns[i] for i in indices[:top_n]]
values = importances[indices[:top_n]]

plt.figure(figsize=(10,8))
plt.barh(range(top_n)[::-1], values, align='center')  # reverse for descending order
plt.yticks(range(top_n)[::-1], features)
plt.xlabel("Feature Importance")
plt.title("Top 20 Hand-Crafted Feature Importances")
plt.tight_layout()
plt.show()