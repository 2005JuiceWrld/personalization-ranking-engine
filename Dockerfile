FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models/two_tower.pt ./models/two_tower.pt
COPY models/two_tower_user_mapping.csv ./models/two_tower_user_mapping.csv
COPY models/two_tower_item_mapping.csv ./models/two_tower_item_mapping.csv
COPY models/two_tower_items_hnsw.faiss ./models/two_tower_items_hnsw.faiss
COPY models/lightgbm_ranker.txt ./models/lightgbm_ranker.txt
COPY models/popularity.csv ./models/popularity.csv

COPY data/features ./data/features
COPY data/processed/train.csv ./data/processed/train.csv

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
