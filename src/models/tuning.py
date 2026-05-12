import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoConfig,
    BertForSequenceClassification,
    Trainer, TrainingArguments,
    EarlyStoppingCallback
)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path: str, text_column: str = "comment_text", label_column: str = "toxic"):
    df = pd.read_csv(data_path)
    if 'toxic' not in df.columns:
        toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        if all(c in df.columns for c in toxic_cols):
            df['toxic'] = (df[toxic_cols].sum(axis=1) > 0).astype(int)
        else:
            raise ValueError("Не удалось найти колонку с метками. Укажите label_column.")
    texts = df[text_column].astype(str).values
    labels = df[label_column].values
    return texts, labels

def train_and_log(data_path: str, model_name: str = "bert-base-uncased", output_dir: str = "models/finetuned"):
    texts, labels = load_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = ToxicityDataset(X_train, y_train, tokenizer)
    val_dataset = ToxicityDataset(X_val, y_val, tokenizer)

    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = BertForSequenceClassification.from_pretrained(model_name, config=config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="mlflow",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='binary')
        return {"accuracy": acc, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    with mlflow.start_run(run_name="toxic_finetune") as run:
        mlflow.log_params({
            "model_name": model_name,
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "max_seq_length": 256
        })

        trainer.train()

        eval_metrics = trainer.evaluate()
        mlflow.log_metrics(eval_metrics)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        signature = infer_signature(X_val[:5], np.argmax(trainer.predict(val_dataset)[0], axis=1))
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="toxic_model",
            task="text-classification",
            signature=signature
        )

        print(f"Model saved to {output_dir} and logged to MLflow run {run.info.run_id}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV with 'comment_text' and 'toxic' columns")
    parser.add_argument("--model", default="bert-base-uncased", help="Base model name")
    parser.add_argument("--output", default="models/finetuned", help="Output directory")
    args = parser.parse_args()
    train_and_log(args.data, args.model, args.output)