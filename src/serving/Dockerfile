FROM python:3.11-slim

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia apenas o necessário para instalar dependências primeiro (cache layer)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e "." 2>/dev/null || pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    onnxruntime \
    numpy \
    pandas \
    scikit-learn \
    joblib \
    yfinance \
    python-dotenv

# Copia o código-fonte
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/processed/artifacts/ ./data/processed/artifacts/

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
