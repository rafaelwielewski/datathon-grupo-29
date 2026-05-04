# Datathon FIAP Fase 05 — Grupo 29

Assistente de previsão de atraso de voos com agente ReAct, RAG e CatBoost.

## Stack

| Camada | Tecnologia |
|---|---|
| Modelo | CatBoost (classificação binária) + calibração Platt |
| Agente | LangGraph ReAct + RAG (FAISS) |
| API | FastAPI |
| Rastreamento | MLflow (traces + experimentos) |
| Monitoramento | Prometheus + Grafana + Evidently (drift) |
| Segurança | Guardrails (prompt injection) + redação de PII |

## Quickstart local

```bash
make setup       # instala dependências e cria pastas
make data        # gera artefatos de serving sem treino completo
make train       # treina CatBoost e salva artefatos
make serve       # API FastAPI em localhost:8000
make mlflow-ui   # MLflow UI em localhost:5000
```

## Quickstart Docker

```bash
make docker-serve   # API + MLflow + Prometheus + Grafana
```

| Serviço | URL |
|---|---|
| API | http://localhost:8000 |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

## Estrutura

```
src/
  agent/      # agente ReAct (LangGraph) + RAG pipeline
  features/   # feature engineering (CatBoost)
  models/     # treino, predictor, baselines
  serving/    # FastAPI app + logging
  monitoring/ # drift detection (Evidently) + métricas Prometheus
  security/   # guardrails de entrada + redação de PII na saída
configs/      # hiperparâmetros, Prometheus, Grafana, agent
data/
  raw/        # CSVs originais (versionados via DVC)
  processed/  # artefatos de serving (model, route_stats, airport_state_map)
  golden_set/ # golden set para avaliação do agente
docs/         # Model Card, System Card, LGPD, OWASP
evaluation/   # RAGAS, LLM-as-judge, A/B prompts
tests/        # pytest (cobertura >= 60%)
scripts/      # generate_data_artifacts.py (make data)
```

## Exemplo de uso

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual a chance de atraso no voo AA de ATL para LAX às 8h?"}'
```

## Qualidade

```bash
make lint        # ruff
make type-check  # mypy
make security    # bandit
make test        # pytest + cobertura HTML
make pre-commit  # todos os hooks
```

## Métricas do modelo

Ver [docs/MODEL_CARD.md](docs/MODEL_CARD.md).
