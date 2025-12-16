"""
Baseline Models for Reddit Data - SVM and Random Forest
Uses same data split as transformer for fair comparison
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import time

print("="*70)
print("TRAINING BASELINE MODELS ON REDDIT DATA")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_parquet("data/reddit_features.parquet")

# Sample same data as transformer for fair comparison
print("Sampling data (same as transformer)...")
df_sampled = df.groupby('label_encoded', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 10000), random_state=42)
).reset_index(drop=True)

print(f"Dataset size: {len(df_sampled)}")
print(f"Label distribution: {df_sampled['label_encoded'].value_counts().to_dict()}")

# Prepare features
y = df_sampled["label_encoded"]
X = df_sampled.drop(columns=["label_encoded", "clean_text", "author"])

# Keep only numeric columns
X = X.select_dtypes(include=['float64', 'int64'])

print(f"\nFeatures: {X.shape[1]} numeric features")

# Split data (same split as transformer)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

results = {}

# ========== SVM ==========
print("\n" + "="*70)
print("TRAINING SVM")
print("="*70)

start_time = time.time()
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42, max_iter=1000)
svm_model.fit(X_train_scaled, y_train)
svm_train_time = time.time() - start_time

y_pred_svm = svm_model.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm, average='weighted')

results['SVM'] = {
    'Accuracy': svm_acc,
    'F1 Score': svm_f1,
    'Training Time (min)': svm_train_time / 60
}

print(f"\nSVM Results:")
print(f"Accuracy: {svm_acc:.4f}")
print(f"F1 Score: {svm_f1:.4f}")
print(f"Training Time: {svm_train_time/60:.2f} minutes")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=["Not Depressed", "Depressed"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# ========== Random Forest ==========
print("\n" + "="*70)
print("TRAINING RANDOM FOREST")
print("="*70)

start_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100,  # Reduced for faster training
    max_depth=20,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train)
rf_train_time = time.time() - start_time

y_pred_rf = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')

results['Random Forest'] = {
    'Accuracy': rf_acc,
    'F1 Score': rf_f1,
    'Training Time (min)': rf_train_time / 60
}

print(f"\nRandom Forest Results:")
print(f"Accuracy: {rf_acc:.4f}")
print(f"F1 Score: {rf_f1:.4f}")
print(f"Training Time: {rf_train_time/60:.2f} minutes")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=["Not Depressed", "Depressed"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# ========== Logistic Regression (already run, but add to results) ==========
# From reddit_LR.py results on full dataset: 89.10% accuracy
# We'll use the same split to get comparable results
print("\n" + "="*70)
print("TRAINING LOGISTIC REGRESSION")
print("="*70)

from sklearn.linear_model import LogisticRegression

start_time = time.time()
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_train_time = time.time() - start_time

y_pred_lr = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr, average='weighted')

results['Logistic Regression'] = {
    'Accuracy': lr_acc,
    'F1 Score': lr_f1,
    'Training Time (min)': lr_train_time / 60
}

print(f"\nLogistic Regression Results:")
print(f"Accuracy: {lr_acc:.4f}")
print(f"F1 Score: {lr_f1:.4f}")
print(f"Training Time: {lr_train_time/60:.2f} minutes")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=["Not Depressed", "Depressed"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# Save results
import json
with open("baseline_models_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("BASELINE MODELS TRAINING COMPLETE")
print("="*70)
print(f"\nResults saved to: baseline_models_results.json")

