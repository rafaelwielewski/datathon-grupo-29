# Datathon FIAP Fase 05 — Grupo 29

Previsão de preço de ações (AAPL) com LSTM para horizonte de 5 dias úteis.

## Quickstart

```bash
make setup       # cria pastas e instala dependências
make train       # treina o modelo e exporta para ONNX
make serve       # sobe a API FastAPI em localhost:8000
make mlflow-ui   # abre o MLflow UI em localhost:5000
```

## Estrutura

```
src/
  features/   # feature engineering (16 indicadores técnicos)
  models/     # LSTM, baselines, pipeline de treino
  serving/    # FastAPI + ONNX runtime
  agent/      # agente ReAct + RAG
  monitoring/ # drift detection + telemetria
  security/   # guardrails + detecção de PII
configs/      # hiperparâmetros externalizados
notebooks/    # EDA exploratória
docs/         # Model Card, System Card, LGPD, OWASP
evaluation/   # RAGAS, LLM-as-judge, A/B prompts
tests/        # pytest (cov >= 60%)
```

## Métricas do modelo (test set)

| Métrica | LSTM | Naive Baseline |
|---|---|---|
| MAE (USD) | 7.36 | 7.31 |
| RMSE (USD) | 9.86 | 9.80 |
| MAPE (%) | 3.27% | 3.26% |
| Directional Accuracy | 52.5% | — |

Ver [docs/MODEL_CARD.md](docs/MODEL_CARD.md) para mapeamento de métricas de negócio.

## Qualidade

```bash
make lint        # ruff
make type-check  # mypy
make security    # bandit
make test        # pytest + cobertura
```
