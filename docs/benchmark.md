# Benchmark — Flight Delay Prediction

## Modelo Principal: CatBoostClassifier + Platt + Prior-Shift

| Métrica | Valor (Test — Meses 11-12) |
|---|---|
| AUC-ROC | ~0.72 |
| Precision | >= 0.66 |
| Recall | ~0.51 |
| F1-Score | ~0.58 |
| Threshold escolhido | ~0.607 |

O threshold foi escolhido no conjunto de validação (meses 9-10) com a constraint `Precision >= 0.66`.

## Baselines

| Modelo | Precision | Recall | F1 |
|---|---|---|---|
| MajorityClassBaseline (tudo ON TIME) | 0.00 | 0.00 | 0.00 |
| PriorRateBaseline (thr=0.18) | ~0.18 | 1.00 | ~0.31 |
| **CatBoost + Platt** | **>= 0.66** | **~0.51** | **~0.58** |

## Interpretação

- **Precision >= 0.66**: quando o modelo prevê atraso, tem 66%+ de chance de estar certo.
- **Recall ~0.51**: captura ~51% de todos os voos que de fato atrasam.
- **Trade-off**: threshold alto favorece precision (menos falsos alarmes) às custas de recall.

## Configuração do Agente (LLM)

Ambiente: GitHub Models API (Azure), acesso via `GITHUB_TOKEN`. Inferência remota, sem GPU local necessária.

| Config | Modelo | Temperatura | RAG | Qualidade média | Recomendação |
|---|---|---|---|---|---|
| A | gpt-4o-mini | 0.0 | Não | 4.0/5 | Fallback sem RAG |
| B | gpt-4o-mini | 0.0 | Sim | 5.0/5 | **Recomendado** |

**Configuração padrão:** B — `gpt-4o-mini` com RAG (FAISS + all-MiniLM-L6-v2) e temperatura 0.0.

## Stack de Embeddings

- Modelo: `all-MiniLM-L6-v2` (sentence-transformers, 100% local)
- Vector store: FAISS (in-memory)
- Chunks: 512 tokens, overlap 50 tokens
- K resultados por query: 4
