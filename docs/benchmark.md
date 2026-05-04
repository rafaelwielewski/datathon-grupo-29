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
| A | gpt-4.1 | 0.0 | Não | 4.8/5 | Fallback sem RAG |
| B | gpt-4.1 | 0.0 | Sim | 4.8/5 | **Recomendado** |
| C | gpt-4o | 0.0 | Sim | 4.5/5 | Alternativa alta performance |

**Configuração padrão:** B — `gpt-4.1` com RAG (FAISS + all-MiniLM-L6-v2) e temperatura 0.0.
**Config C:** Variação utilizando o modelo `gpt-4o` testada como alternativa para comparar resultados do LLM-as-a-Judge.

## Stack de Embeddings

- Modelo: `all-MiniLM-L6-v2` (sentence-transformers, 100% local)
- Vector store: FAISS (in-memory)
- Chunks: 512 tokens, overlap 50 tokens
- K resultados por query: 4

## Governança e Deploy (Approval Gate)

Para garantir qualidade e segurança, o ciclo de vida dos modelos de ML utiliza um processo de **Approval Gate** integrado ao MLflow.

O pipeline de treinamento registra o modelo e suas métricas continuamente, mas sua promoção à produção é condicional. O modelo só é promovido e marcado formalmente com `approval_status=approved` se a variável de ambiente `MLFLOW_APPROVE=true` estiver definida (tipicamente via pipeline de CI/CD).
Esse processo garante uma etapa de revisão validada (seja por métricas estabelecidas ou por um _Human-in-the-Loop_) antes de disponibilizar uma nova versão de modelo para o agente consumi-lo.
O pipeline de treinamento registra o modelo e suas métricas continuamente, mas a aprovação automática não o promove diretamente à produção. Quando a variável de ambiente `MLFLOW_APPROVE=true` está definida (tipicamente via pipeline de CI/CD), o modelo é marcado formalmente com `approval_status=approved` e transicionado para `Staging`.
Esse processo garante uma etapa de revisão validada (seja por métricas estabelecidas ou por um _Human-in-the-Loop_) antes de uma promoção posterior para produção. **Importante:** no estado atual, essa aprovação altera metadados/estágio no registry, mas não muda automaticamente o modelo servido pelo agente, que ainda consome artefatos locais.
