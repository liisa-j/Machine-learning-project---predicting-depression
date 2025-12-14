import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


df = pd.read_parquet("data/reddit_features.parquet")


y = df["label_encoded"]
X = df.drop(columns=["label_encoded", "clean_text"])  


non_numeric = X.select_dtypes(exclude='number').columns.tolist()
if non_numeric:
    print("Warning: non-numeric columns detected, they need encoding or dropping:", non_numeric)
    X = X.drop(columns=non_numeric)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)


y_pred = clf.predict(X_test_scaled)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))