import os
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


PARQUET_FILE = "/gpfs/helios/home/kerda8/shorty_clean.parquet"
MODEL_NAME = "mental/mental-bert-base-uncased"
MAX_LENGTH = 256
SEED = 42


print("Loading parquet dataset...")
df = pd.read_parquet(PARQUET_FILE)


df = df.rename(columns={"clean_text": "post", "label_encoded": "label"})

df = df[["post", "label"]].dropna()

print("Dataset sizes:")
print(df["label"].value_counts())


label2id = {0: 0, 1: 1}  
id2label = {0: "class_0", 1: "class_1"}  


hf_dataset = Dataset.from_pandas(df)
dataset = hf_dataset.train_test_split(test_size=0.2, seed=SEED)


print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,)


def tokenize_function(examples):
    return tokenizer(
        examples["post"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"],)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary")
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,}

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,    
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    seed=SEED,
    logging_steps=100,
    save_total_limit=2,)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,)

print("Starting training...")
trainer.train()

print("Evaluating...")
results = trainer.evaluate()
print(results)


trainer.save_model("mentalbert_twitter_classifier")
tokenizer.save_pretrained("mentalbert_twitter_classifier")

print("Training complete.")