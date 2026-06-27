import pandas as pd
import argparse
from pathlib import Path
import mlflow
import logging
from typing import List, Dict
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path: str):
    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)

    if 'text' not in df.columns:
        raise ValueError("Dataset must contain 'text' column")

    logger.info(f"Loaded {len(df)} samples")
    return df


def load_model_from_registry(model, model_name: str, version: str = None):
    if version:
        model_uri = f"models:/{model_name}/{version}"
    else:
        model_uri = f"models:/{model_name}/latest"

    logger.info(f"Loading model from {model_uri}")
    loaded = model.load_from_mlflow(model_uri)

    if not loaded:
        raise RuntimeError(f"Failed to load model from {model_uri}")

    logger.info("Model loaded successfully")
    return model


def predict_batch(model, texts: List[str], threshold: float = 0.5, batch_size: int = 32):
    logger.info(f"Making predictions for {len(texts)} texts with threshold {threshold}")

    results = model.predict_batch(texts, batch_size=batch_size, threshold=threshold)

    return results


def save_predictions(results: List[Dict], output_path: str):
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

    return df


def generate_report(df: pd.DataFrame):
    report = {
        "total_samples": len(df),
        "toxic_count": len(df[df['label'] == 'toxic']),
        "non_toxic_count": len(df[df['label'] == 'non-toxic']),
        "toxic_rate": len(df[df['label'] == 'toxic']) / len(df) if len(df) > 0 else 0,
        "avg_confidence_toxic": df[df['label'] == 'toxic']['confidence'].mean() if len(
            df[df['label'] == 'toxic']) > 0 else 0,
        "avg_confidence_non_toxic": df[df['label'] == 'non-toxic']['confidence'].mean() if len(
            df[df['label'] == 'non-toxic']) > 0 else 0,
        "avg_toxic_prob": df['toxic_prob'].mean() if len(df) > 0 else 0,
        "min_toxic_prob": df['toxic_prob'].min() if len(df) > 0 else 0,
        "max_toxic_prob": df['toxic_prob'].max() if len(df) > 0 else 0,
    }

    logger.info("=" * 60)
    logger.info("Prediction Report:")
    for key, value in report.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=" * 60)

    return report


def save_report_to_file(report: Dict, output_path: str):
    report_df = pd.DataFrame([report])
    report_path = output_path.replace('.csv', '_report.csv')
    report_df.to_csv(report_path, index=False)
    logger.info(f"Report saved to {report_path}")
    return report_path


def init_model(use_mlflow=True, tracking_uri=None, experiment_name="toxicity_model"):
    from model import create_model_with_mlflow, init_mlflow

    if tracking_uri:
        init_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)

    return create_model_with_mlflow(use_mlflow=use_mlflow, experiment_name=experiment_name)


def log_to_mlflow(metrics: Dict):
    try:
        active_run = mlflow.active_run()
        if active_run is None:
            logger.warning("No active MLflow run. Starting a new run...")
            mlflow.start_run(run_name="prediction_logging", nested=True)

        mlflow.log_metrics(metrics)
        logger.info(f"Logged {len(metrics)} metrics to MLflow")
        return True
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Predict toxicity on dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file with data to predict')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to save predictions')
    parser.add_argument('--text-column', type=str, default='text', help='Column name for text')
    parser.add_argument('--threshold', type=float, default=0.5, help='Toxicity threshold')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for predictions')
    parser.add_argument('--experiment-name', type=str, default='toxicity_predictions', help='MLflow experiment name')
    parser.add_argument('--tracking-uri', type=str, default=None, help='MLflow tracking URI')

    parser.add_argument('--model-name', type=str, default=None, help='Model name from MLflow Registry')
    parser.add_argument('--model-version', type=str, default=None, help='Model version from MLflow Registry')
    parser.add_argument('--model-uri', type=str, default=None, help='Direct model URI (overrides model-name)')

    parser.add_argument('--use-mlflow', action='store_true', help='Use MLflow tracking')
    parser.add_argument('--log-predictions', action='store_true', help='Log predictions to MLflow as artifacts')
    parser.add_argument('--report', action='store_true', help='Generate and log report')
    parser.add_argument('--id-column', type=str, default=None, help='Column with IDs for tracking')
    parser.add_argument('--save-report', action='store_true', help='Save report to file')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting prediction pipeline")
    logger.info(f"Arguments: {args}")
    logger.info("=" * 60)

    local_model = init_model(
        use_mlflow=args.use_mlflow,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name
    )

    if args.use_mlflow:
        mlflow.start_run(run_name="prediction_batch")
        mlflow.log_params({
            "data_path": args.data,
            "threshold": args.threshold,
            "batch_size": args.batch_size,
            "model_name": args.model_name,
            "model_version": args.model_version,
        })
        logger.info(f"MLflow run started: {mlflow.active_run().info.run_id}")

    df = load_dataset(args.data)

    if args.model_uri:
        logger.info(f"Loading model from URI: {args.model_uri}")
        loaded = local_model.load_from_mlflow(args.model_uri)
        if not loaded:
            raise RuntimeError(f"Failed to load model from {args.model_uri}")
    elif args.model_name:
        load_model_from_registry(local_model, args.model_name, args.model_version)
    else:
        logger.warning("No model specified, using currently loaded model")

    texts = df[args.text_column].tolist()

    logger.info("Making predictions...")
    results = predict_batch(local_model, texts, threshold=args.threshold, batch_size=args.batch_size)

    logger.info("Saving predictions...")
    result_df = save_predictions(results, args.output)

    if args.id_column and args.id_column in df.columns:
        ids = df[args.id_column].tolist()
        result_df.insert(0, 'id', ids)
        result_df.to_csv(args.output, index=False)

    if args.log_predictions and args.use_mlflow:
        mlflow.log_artifact(args.output)

    report = generate_report(result_df)

    metrics = {
        "toxic_count": report['toxic_count'],
        "non_toxic_count": report['non_toxic_count'],
        "toxic_rate": report['toxic_rate'],
        "avg_confidence_toxic": report['avg_confidence_toxic'],
        "avg_confidence_non_toxic": report['avg_confidence_non_toxic'],
        "avg_toxic_prob": report['avg_toxic_prob'],
        "min_toxic_prob": report['min_toxic_prob'],
        "max_toxic_prob": report['max_toxic_prob']
    }
    log_to_mlflow(metrics)

    if args.save_report:
        save_report_to_file(report, args.output)

    logger.info("=" * 60)
    logger.info("Prediction pipeline finished")

    if args.use_mlflow:
        mlflow.end_run()
        logger.info("MLflow run ended")

    local_model.end_mlflow_run()


if __name__ == "__main__":
    main()
