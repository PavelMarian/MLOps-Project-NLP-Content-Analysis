from fastapi import APIRouter, UploadFile, File, HTTPException, Query
import pandas as pd
import io
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from db import Session, Prediction

router = APIRouter(prefix="/data", tags=["data_loader"])
logger = logging.getLogger(__name__)

def normalize_label(value) -> str | None:

    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        if value == 1:
            return 'toxic'
        elif value == 0:
            return 'non-toxic'
        else:
            return None
    s = str(value).strip().lower()
    if s in ('toxic', '1', 'true', 'yes', 'positive'):
        return 'toxic'
    elif s in ('not toxic', 'non-toxic', '0', 'false', 'no', 'negative', 'not toxic'):
        return 'non-toxic'
    else:
        return None

@router.post("/load-new-data")
async def load_labeled_data(
    file: UploadFile = File(..., description="CSV файл с данными"),
    clear_existing: bool = Query(False, description="Удалить предыдущие размеченные записи перед загрузкой"),
    text_column: str = Query("text", description="Имя колонки с текстом"),
    label_column: str = Query("label", description="Имя колонки с меткой (0/1 или toxic/not toxic)")
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {str(e)}")

    if text_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Колонка '{text_column}' не найдена")
    if label_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Колонка '{label_column}' не найдена")

    df['_normalized_label'] = df[label_column].apply(normalize_label)
    invalid = df[df['_normalized_label'].isna()]
    if not invalid.empty:
        bad_values = invalid[label_column].unique().tolist()
        raise HTTPException(
            status_code=400,
            detail=f"Обнаружены неподдерживаемые метки: {bad_values}. "
                   f"Используйте 0/1 или 'toxic'/'not toxic'."
        )

    session = Session()
    try:
        if clear_existing:
            deleted = session.query(Prediction).filter(Prediction.feedback == True).delete()
            logger.info(f"Удалено {deleted} старых размеченных записей")
            session.commit()

        batch_size = 1000
        total = len(df)
        inserted = 0

        for start in range(0, total, batch_size):
            batch = df.iloc[start:start + batch_size]
            records = []
            for _, row in batch.iterrows():
                label = row['_normalized_label']  # 'toxic' или 'non-toxic'
                toxic_prob = 1.0 if label == 'toxic' else 0.0
                records.append(Prediction(
                    text=str(row[text_column]),
                    cleaned_text=str(row[text_column]),  # можно добавить очистку позже
                    toxic_prob=toxic_prob,
                    label=label,
                    confidence=1.0,
                    threshold_used=0.5,
                    feedback=True
                ))
            session.add_all(records)
            session.commit()
            inserted += len(records)
            logger.info(f"Вставлено {inserted} из {total} записей")

    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении в БД: {str(e)}")
    finally:
        session.close()

    return {
        "status": "ok",
        "inserted": inserted,
        "clear_existing": clear_existing,
        "columns": df.columns.tolist(),
        "sample": df.head(3).to_dict(orient="records")
    }
