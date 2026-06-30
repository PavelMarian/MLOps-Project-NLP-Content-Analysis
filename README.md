# Toxicity Detection MLOps Platform

Платформа для детекции токсичности текстов с полным циклом MLOps: инференс, сбор обратной связи, мониторинг дрейфа (data drift, target drift, concept drift), переобучение и расчет метрик

---

## Основные возможности

- Система мониторинга дрейфа (Data / Target / Concept Drift) с использованием Prometheus и Grafana
- Ручное переобучение модели на основе накопленных пользовательских фидбеков
- Встроенный веб-интерфейс для визуализации статуса системы, графиков и логов
- Трекинг в MLFlow

---

## Основные модули

| Компонент | Описание |
| :--- | :--- |
| **`api.py`** | Основной файл FastAPI. Содержит эндпоинты для предсказаний, сбора обратной связи (ручной разметки ответа), метрик, моделей и веб-страниц |
| **`db.py`** | Инициализация SQLite базы данных |
| **`drift_router.py`** | Эндпоинты для детектора дрейфа. Содержит логику инициализации детектора, запуска проверок и получения истории |
| **`load_new_data.py`** | Эндпоинты для загрузки входных размеченных данных в формате csv (симуляция накопления и ручной разметки текстов за некоторый период) |
| **`load_ref_data.py`** | Скрипт для загрузки референсного датасета в формате csv |
| **`drift_detector.py`** | Реализует классы для обнаружения дрейфа данных (Data Drift), дрейфа целевой переменной (Target Drift) и дрейфа концеции(Concept Drift) |
| **`prometheus_client.py`** | Сбор метрик для Prometheus |

---

## Модель и данные

В проекте используется предобученная модель для классификации токсичности `s-nlp/russian_toxicity_classifier` (https://huggingface.co/s-nlp/russian_toxicity_classifier). Исходная модель была обучена на двух датасетах: Russian Language Toxic Comments (https://www.kaggle.com/blackmoon/russian-language-toxic-comments/metadata) и Toxic Russian Comments (https://www.kaggle.com/alexandersemiletov/toxic-russian-comments)

---

## Запуск

Клонируйте репозиторий
```
git clone https://github.com/PavelMarian/MLOps-Project-NLP-Content-Analysis
```

Первый вариант: сборка docker-образа
```
docker-compose up -d --build
```

Второй вариант: ручной запуск
```
pip install -r requirements.txt
python api.py
```

Приложение будет доступно по адресу `http://localhost:8000`
