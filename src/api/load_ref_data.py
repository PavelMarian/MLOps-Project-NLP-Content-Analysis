import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
import logging

sys.path.append(str(Path(__file__).parent.parent))

from db import Session, ReferenceData, engine, Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_reference_dataset(
    csv_path: str,
    text_column: str = "text",
    label_column: int = "label",
    batch_size: int = 1000,
    clear_existing: bool = False,
):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Loading reference dataset from {csv_path}")
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found")
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found")

    df[text_column] = df[text_column].astype(str)
    df[label_column] = df[label_column].astype(int)

    session = Session()
    try:
        if clear_existing:
            logger.info("Clearing existing reference data...")
            session.query(ReferenceData).delete()
            session.commit()

        total = len(df)
        logger.info(f"Inserting {total} records...")
        inserted = 0
        for i in range(0, total, batch_size):
            batch = df.iloc[i:i+batch_size]
            records = []
            for _, row in batch.iterrows():
                records.append(
                    ReferenceData(
                        text=row[text_column],
                        label=row[label_column],
                    )
                )
            session.add_all(records)
            session.commit()
            inserted += len(records)
            logger.info(f"Inserted {inserted}/{total} records")

        logger.info(f"Reference dataset loaded successfully. Total: {inserted}")

    except Exception as e:
        session.rollback()
        logger.error(f"Error loading reference data: {e}")
        raise
    finally:
        session.close()


def create_reference_table():
    Base.metadata.create_all(engine)
    logger.info("ReferenceData table created/verified")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load reference dataset into DB")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--text-col", default="text", help="Text column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--clear", action="store_true", help="Clear existing reference data before loading")
    parser.add_argument("--batch", type=int, default=1000, help="Batch size for insertion")
    args = parser.parse_args()

    create_reference_table()
    load_reference_dataset(
        csv_path=args.csv,
        text_column=args.text_col,
        label_column=args.label_col,
        batch_size=args.batch,
        clear_existing=args.clear,
    )
