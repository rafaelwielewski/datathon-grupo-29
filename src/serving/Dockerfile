FROM python:3.11-slim

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia pyproject.toml para instalação de dependências
COPY pyproject.toml ./
RUN pip install --no-cache-dir --default-timeout=100 --retries=10 .

# Copia o código-fonte
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/processed/artifacts/ ./data/processed/artifacts/
COPY .env ./

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
