"""
Transformer Model for Depression Prediction - Fast Training Version
Uses DistilBERT with optimized settings for faster training
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import time
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
print("Loading data...")
df = pd.read_parquet("data/reddit_features.parquet")

# Sample data for faster training (use full data by removing .sample())
print("Sampling data for faster training...")
df_sampled = df.groupby('label_encoded', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 10000), random_state=42)
).reset_index(drop=True)

# Use text and labels
texts = df_sampled["clean_text"].astype(str).tolist()
labels = df_sampled["label_encoded"].astype(int).tolist()

print(f"Dataset size: {len(texts)}")
print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Further split train into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# Initialize tokenizer and model
model_name = "distilbert-base-uncased"
print(f"\nLoading {model_name}...")
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
model.to(device)

# Dataset class
class DepressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):  # Shorter sequences
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Create datasets
print("\nTokenizing data...")
train_dataset = DepressionDataset(X_train, y_train, tokenizer)
val_dataset = DepressionDataset(X_val, y_val, tokenizer)
test_dataset = DepressionDataset(X_test, y_test, tokenizer)

# Training arguments - optimized for speed
training_args = TrainingArguments(
    output_dir="./transformer_results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./transformer_logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,
    learning_rate=2e-5,
    fp16=False,
    dataloader_num_workers=0,
    report_to="none",
    remove_unused_columns=False,
)

# Metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {
        "accuracy": accuracy,
        "f1": f1
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

# Train
print("\n" + "="*60)
print("TRAINING TRANSFORMER MODEL")
print("="*60)
start_time = time.time()

trainer.train()

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time/60:.2f} minutes")

# Evaluate on test set
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

test_results = trainer.evaluate(test_dataset)
print(f"\nTest Results:")
print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"F1 Score: {test_results['eval_f1']:.4f}")

# Get predictions for detailed metrics
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Depressed", "Depressed"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate additional metrics
accuracy = accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average="weighted")
f1_macro = f1_score(y_test, y_pred, average="macro")

print("\n" + "="*60)
print("SUMMARY METRICS")
print("="*60)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")

# Save model
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)
model_save_path = "./transformer_model"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

# Save results to file
results_summary = f"""
TRANSFORMER MODEL RESULTS
========================

Model: DistilBERT-base-uncased
Dataset: Reddit Depression Classification
Training Samples: {len(X_train)}
Validation Samples: {len(X_val)}
Test Samples: {len(X_test)}

RESULTS:
--------
Accuracy: {accuracy:.4f}
F1 Score (Weighted): {f1_weighted:.4f}
F1 Score (Macro): {f1_macro:.4f}
Training Time: {training_time/60:.2f} minutes

CONFUSION MATRIX:
{confusion_matrix(y_test, y_pred)}
"""

with open("transformer_results_summary.txt", "w") as f:
    f.write(results_summary)

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"\nFinal Test Accuracy: {accuracy:.4f}")
print(f"Final Test F1 Score (Weighted): {f1_weighted:.4f}")
print(f"Training Time: {training_time/60:.2f} minutes")
print(f"\nResults saved to: transformer_results_summary.txt")

