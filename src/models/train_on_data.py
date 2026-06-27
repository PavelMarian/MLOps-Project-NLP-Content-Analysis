import pandas as pd
import argparse
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from model import model, init_mlflow
import mlflow
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path: str):
    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)

    required_columns = ['text', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")

    return df


def prepare_data(df, text_column='text', label_column='label', test_size=0.2, random_state=42):
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()

    return train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )


def validate_labels(labels):
    unique_labels = set(labels)
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Labels must be 0 or 1. Found: {unique_labels}")
    logger.info(f"Valid labels found: {unique_labels}")


def main():
    parser = argparse.ArgumentParser(description='Train toxicity model on dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file with data')
    parser.add_argument('--text-column', type=str, default='text', help='Column name for text')
    parser.add_argument('--label-column', type=str, default='label', help='Column name for label')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='models/trained', help='Output directory')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split size')
    parser.add_argument('--experiment-name', type=str, default='toxicity_training', help='MLflow experiment name')
    parser.add_argument('--tracking-uri', type=str, default=None, help='MLflow tracking URI')
    parser.add_argument('--register-model', action='store_true', help='Register model in MLflow Registry')
    parser.add_argument('--model-name', type=str, default='toxicity_classifier', help='Model name for registration')
    parser.add_argument('--use-mlflow', action='store_true', default=True, help='Use MLflow for tracking')

    args = parser.parse_args()

    if args.tracking_uri:
        init_mlflow(tracking_uri=args.tracking_uri, experiment_name=args.experiment_name)

    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info(f"Arguments: {args}")
    logger.info("=" * 60)

    df = load_dataset(args.data)

    logger.info("Validating labels...")
    validate_labels(df[args.label_column].tolist())

    logger.info("Splitting data into train and validation sets...")
    train_texts, val_texts, train_labels, val_labels = prepare_data(
        df,
        text_column=args.text_column,
        label_column=args.label_column,
        test_size=args.test_size
    )

    logger.info(f"Train size: {len(train_texts)}")
    logger.info(f"Validation size: {len(val_texts)}")

    logger.info("Starting model training...")
    training_result = model.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        save_to_mlflow=args.use_mlflow
    )

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Results: {training_result}")

    if args.register_model and training_result.get('success', False):
        logger.info("Registering model in MLflow Registry...")
        registered = model.register_model(model_name=args.model_name)
        if registered:
            logger.info(f"Model registered successfully: {args.model_name} (version {registered.version})")

    logger.info("=" * 60)
    logger.info("Training pipeline finished")
    model.end_mlflow_run()


if __name__ == "__main__":
    main()
