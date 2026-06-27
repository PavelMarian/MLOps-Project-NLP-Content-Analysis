import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import re
from typing import List, Dict, Union, Optional, Tuple
import mlflow
import mlflow.pytorch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from mlflow.models import infer_signature
import mlflow.transformers
import numpy as np


class ToxicityModel:
    def __init__(self,
                 model_path="s-nlp/russian_toxicity_classifier",
                 device=None,
                 threshold=0.5,
                 use_mlflow=True,
                 experiment_name="toxicity_model"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.model_path = model_path
        self.use_mlflow = use_mlflow
        self.mlflow_run = None

        if self.use_mlflow:
            mlflow.set_experiment(experiment_name)
            self.mlflow_run = mlflow.start_run(run_name=f"load_{model_path.replace('/', '_')}")
            mlflow.log_params({
                "model_path": model_path,
                "device": self.device,
                "threshold": threshold,
                "model_type": "BertForSequenceClassification"
            })

        print(f"Loading model from {model_path}...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

            if self.use_mlflow:
                num_params = sum(p.numel() for p in self.model.parameters())
                mlflow.log_metrics({
                    "num_parameters": num_params,
                    "vocab_size": self.tokenizer.vocab_size
                })
                print(f"MLflow tracking started - Run ID: {self.mlflow_run.info.run_id}")

        except Exception as e:
            if self.use_mlflow and self.mlflow_run:
                try:
                    mlflow.end_run(status="FAILED")
                except Exception:
                    pass
            raise e

        print(f"Model loaded on {self.device}")

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        text = text.lower()
        text = re.sub(r'http\S+|@\w+|#\w+', '', text)
        text = re.sub(r'[^а-яё\s\.\!\?\,]', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @torch.no_grad()
    def predict(self, texts: Union[str, List[str]], threshold: Optional[float] = None) -> Union[Dict, List[Dict]]:
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        thr = threshold or self.threshold
        cleaned_texts = [self.clean_text(t) for t in texts]

        inputs = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        probs = F.softmax(logits, dim=-1)
        toxic_probs = probs[:, 1].cpu().numpy()

        results = []
        for text, cleaned, prob in zip(texts, cleaned_texts, toxic_probs):
            result = {
                "text": text,
                "cleaned_text": cleaned,
                "toxic_prob": float(prob),
                "label": "toxic" if prob > thr else "non-toxic",
                "confidence": float(max(prob, 1 - prob))
            }
            results.append(result)

        if self.use_mlflow and single_input:
            mlflow.log_metrics({
                "prediction_toxic_prob": float(toxic_probs[0]),
                "prediction_confidence": float(max(toxic_probs[0], 1 - toxic_probs[0]))
            })

        return results[0] if single_input else results

    def predict_batch(self, texts: List[str], batch_size: int = 32, threshold: Optional[float] = None) -> List[Dict]:
        all_results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = self.predict(batch, threshold)
            all_results.extend(results)
        return all_results

    def get_toxic_score(self, text: str) -> float:
        result = self.predict(text)
        return result["toxic_prob"]

    def is_toxic(self, text: str, threshold: Optional[float] = None) -> bool:
        thr = threshold or self.threshold
        return self.get_toxic_score(text) > thr

    def get_model_info(self) -> Dict:
        info = {
            "model_path": self.model_path,
            "model_type": "BertForSequenceClassification",
            "device": self.device,
            "vocab_size": self.tokenizer.vocab_size,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "threshold": self.threshold
        }
        if self.use_mlflow and self.mlflow_run:
            info["mlflow_run_id"] = self.mlflow_run.info.run_id
            info["mlflow_experiment_id"] = self.mlflow_run.info.experiment_id
        return info

    def _tokenize(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=512
        )

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='binary'),
            'precision': precision_score(labels, predictions, average='binary'),
            'recall': recall_score(labels, predictions, average='binary')
        }

    def train(self,
              train_texts: List[str],
              train_labels: List[int],
              val_texts: Optional[List[str]] = None,
              val_labels: Optional[List[int]] = None,
              epochs: int = 3,
              batch_size: int = 16,
              learning_rate: float = 2e-5,
              output_dir: str = "models/finetuned",
              save_to_mlflow: bool = True) -> Dict:

        if len(train_texts) < 10:
            print("Not enough data for training. Need at least 10 examples.")
            return {"success": False, "error": "Not enough data"}

        print(f"Starting training with {len(train_texts)} examples...")

        if self.use_mlflow and save_to_mlflow:
            with mlflow.start_run(run_name="training", nested=True):
                return self._train_internal(
                    train_texts, train_labels,
                    val_texts, val_labels,
                    epochs, batch_size, learning_rate,
                    output_dir, save_to_mlflow
                )
        else:
            return self._train_internal(
                train_texts, train_labels,
                val_texts, val_labels,
                epochs, batch_size, learning_rate,
                output_dir, save_to_mlflow
            )

    def _train_internal(self,
                        train_texts: List[str],
                        train_labels: List[int],
                        val_texts: Optional[List[str]],
                        val_labels: Optional[List[int]],
                        epochs: int,
                        batch_size: int,
                        learning_rate: float,
                        output_dir: str,
                        save_to_mlflow: bool) -> Dict:

        if self.use_mlflow:
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "train_size": len(train_texts),
                "val_size": len(val_texts) if val_texts else 0
            })

        train_data = {"text": train_texts, "label": train_labels}
        train_dataset = Dataset.from_dict(train_data)
        train_dataset = train_dataset.map(self._tokenize, batched=True)

        has_validation = val_texts is not None and val_labels is not None and len(val_texts) > 0

        if has_validation:
            val_data = {"text": val_texts, "label": val_labels}
            val_dataset = Dataset.from_dict(val_data)
            val_dataset = val_dataset.map(self._tokenize, batched=True)
        else:
            val_dataset = None

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        if has_validation:
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="f1"
            )
        else:
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                eval_strategy="no",
                save_strategy="epoch",
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=False
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self._compute_metrics if has_validation else None,
        )

        trainer.train()

        metrics = {}

        if has_validation:
            eval_results = trainer.evaluate()
            metrics = {
                'eval_accuracy': eval_results.get('eval_accuracy', 0),
                'eval_f1': eval_results.get('eval_f1', 0),
                'eval_precision': eval_results.get('eval_precision', 0),
                'eval_recall': eval_results.get('eval_recall', 0),
                'eval_loss': eval_results.get('eval_loss', 0)
            }

            if self.use_mlflow:
                mlflow.log_metrics(metrics)

        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        if self.use_mlflow and save_to_mlflow:
            mlflow.transformers.log_model(
                transformers_model={
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                },
                artifact_path="model",
            )

        print(f"Training completed. Model saved to {output_dir}")
        metrics["success"] = True
        metrics["model_path"] = output_dir

        return metrics

    def fine_tune(self,
                  feedback_data: List[Dict],
                  epochs: int = 3,
                  batch_size: int = 16,
                  learning_rate: float = 2e-5,
                  output_dir: str = "models/finetuned"):

        if len(feedback_data) < 50:
            print(f"Not enough feedback data for fine-tuning. Need at least 50, got {len(feedback_data)}")
            return {"success": False, "error": "Not enough feedback data"}

        texts = [item['text'] for item in feedback_data]
        labels = [item['correct_label'] for item in feedback_data]

        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        return self.train(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir,
            save_to_mlflow=True
        )

    def save_model_to_mlflow(self, path: str = "models/toxic_model"):
        if not self.use_mlflow:
            print("MLflow is disabled. Enable use_mlflow=True to save.")
            return None

        Path(path).mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        mlflow.transformers.log_model(
            transformers_model={
                "model": self.model,
                "tokenizer": self.tokenizer,
            },
            artifact_path="model",
        )

        print(f"Model saved to MLflow: {path}")
        if self.mlflow_run:
            return self.mlflow_run.info.run_id
        return None

    def register_model(self, model_name: str = "toxicity_classifier"):
        if not self.use_mlflow:
            print("MLflow is disabled. Enable use_mlflow=True to register.")
            return None

        if not self.mlflow_run:
            print("No active MLflow run. Please load or train a model first.")
            return None

        model_uri = f"runs:/{self.mlflow_run.info.run_id}/model"
        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            print(f"Model registered: {model_name} (version {registered_model.version})")
            return registered_model
        except Exception as e:
            print(f"Error registering model: {e}")
            return None

    def load_from_mlflow(self, model_uri: str):
        try:
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded from MLflow: {model_uri}")
            return True
        except Exception as e:
            print(f"Error loading model from MLflow: {e}")
            return False

    def end_mlflow_run(self):
        if self.use_mlflow and self.mlflow_run:
            try:
                mlflow.end_run()
                self.mlflow_run = None
            except Exception:
                pass

    def __del__(self):
        try:
            if self.use_mlflow and self.mlflow_run is not None:
                mlflow.end_run()
        except Exception:
            pass


def init_mlflow(tracking_uri: str = None, experiment_name: str = "toxicity_model"):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    print(f"MLflow initialized with tracking URI: {mlflow.get_tracking_uri()}")


def create_model_with_mlflow(use_mlflow=True, experiment_name="toxicity_model"):
    if use_mlflow:
        init_mlflow(experiment_name=experiment_name)

    return ToxicityModel(
        model_path="s-nlp/russian_toxicity_classifier",
        threshold=0.5,
        use_mlflow=use_mlflow,
        experiment_name=experiment_name
    )


USE_MLFLOW = True
model = create_model_with_mlflow(use_mlflow=USE_MLFLOW)

if __name__ == "__main__":
    test_texts = [
        "ты супер",
        "ты идиот, пошел вон",
        "спасибо за помощь, очень приятно",
        "какой же ты дурак",
        "отличная погода сегодня"
    ]

    print("=" * 60)
    print("Testing Russian Toxicity Classifier with MLflow")
    print("=" * 60)

    for text in test_texts:
        result = model.predict(text)
        print(f"\nText: {text}")
        print(f"Toxic probability: {result['toxic_prob']:.4f}")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")

    print("\n" + "=" * 60)
    print("Testing training functionality")
    print("=" * 60)

    train_texts = [
        "это отличный день",
        "ты молодец, хорошо поработал",
        "спасибо за помощь",
        "я очень рад",
        "прекрасная погода",
        "ты идиот",
        "пошел вон отсюда",
        "какой же ты глупый",
        "ты никчемный человек",
        "ненавижу тебя",
        "отличная работа, продолжай в том же духе",
        "ты очень умный",
        "это было замечательно",
        "спасибо большое",
        "ты лучший друг",
        "как ты мог так поступить",
        "ты меня разочаровал",
        "это ужасно",
        "я в шоке от твоего поведения",
        "ты должен извиниться"
    ]

    train_labels = [
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ]

    val_texts = [
        "хорошая работа",
        "ты дурак",
        "отлично справился",
        "пошел прочь",
        "мне нравится твой подход"
    ]

    val_labels = [0, 1, 0, 1, 0]

    print(f"Training data size: {len(train_texts)}")
    print(f"Validation data size: {len(val_texts)}")

    training_result = model.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=1,
        batch_size=4,
        learning_rate=2e-5,
        output_dir="models/test_training",
        save_to_mlflow=True
    )

    print("\nTraining results:")
    for key, value in training_result.items():
        print(f"{key}: {value}")

    test_after_training = [
        "ты гений",
        "ты полный идиот"
    ]

    print("\n" + "=" * 60)
    print("Testing model after training")
    print("=" * 60)

    for text in test_after_training:
        result = model.predict(text)
        print(f"\nText: {text}")
        print(f"Toxic probability: {result['toxic_prob']:.4f}")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")

    print("\n" + "=" * 60)
    print("Saving model to MLflow...")
    run_id = model.save_model_to_mlflow()

    print("\nRegistering model...")
    registered = model.register_model()

    print("\n" + "=" * 60)
    print("Model Info:", model.get_model_info())

    model.end_mlflow_run()
