## This script runs Random Forest on Twitter data (tweet level) and only extracted features
## Performs awful
##  Inputs shorty_features.parquet

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt

df = pd.read_parquet("data/shorty_features.parquet")

X = df.drop(columns=['user_id', 'tweet_date', 'language', 'clean_text', 'label_encoded'])
y = df['label_encoded']

print(f"Total rows: {df.shape[0]}, Total features: {X.shape[1]}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#to save ram
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)


rf = RandomForestClassifier(
    n_estimators=200,   
    max_depth=None, 
    n_jobs=-1, 
    random_state=42,
    verbose=1
)

print("Training Random Forest...")
start_time = time.time()
rf.fit(X_train, y_train)
end_time = time.time()

# Predict
y_pred = rf.predict(X_test)

# Metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nTotal runtime: {(end_time - start_time)/60:.2f} minutes")


#features importance
fi_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)
print("\nTop 20 Important Features:")
print(fi_df.head(20))
fi_df.to_csv("rf_feature_importance.csv", index=False)

fi_df.head(20).plot(kind="barh", x="feature", y="importance", figsize=(8,6), legend=False)
plt.gca().invert_yaxis()
plt.title("Top 20 Random Forest Feature Importances")
plt.tight_layout()
plt.show()