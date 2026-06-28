FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
ENV PIP_DEFAULT_TIMEOUT=10000
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install 'accelerate>=1.1.0'
RUN pip install 'torchvision'

COPY . .

RUN mkdir -p data mlruns

EXPOSE 8000

CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]