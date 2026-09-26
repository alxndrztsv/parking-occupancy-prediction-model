# Python 3.14 is not optional here: catboost ships cp314 wheels, and the saved
# model bundles are pickles that must be read by the same library versions that
# wrote them. That is the whole point of the image.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 is the OpenMP runtime that LightGBM and XGBoost link against; the slim
# base image does not ship it, and both fail at import without it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so editing code does not invalidate this layer.
# Both files are needed: requirements-dev.txt references requirements.txt.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY conftest.py ./
COPY src/ ./src/
COPY tests/ ./tests/

# configs/ is copied for a standalone image and mounted read-only by compose, so
# hyperparameters can be changed without a rebuild.
COPY configs/ ./configs/

# Inputs and artifacts are mounted at runtime, never baked into the image.
RUN mkdir -p data/raw data/normalized data/reference data/enriched \
             data/processed data/splits models reports

CMD ["python", "-m", "src.pipeline"]
