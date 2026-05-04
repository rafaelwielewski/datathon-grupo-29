# Documento Auxiliar de Documentacao - Datathon Grupo 29

## 1. Visao geral
Este documento resume a solucao desenvolvida no projeto Datathon Grupo 29, cobrindo dados, pipeline de ML, API, agente LLM, RAG, avaliacao, observabilidade, seguranca e governanca. O objetivo e servir como base para gerar um arquivo .docx final.

## 2. Problema e objetivo
O projeto visa prever atrasos de voos domesticos nos EUA (>= 15 minutos) e disponibilizar uma interface em linguagem natural para consultas e explicacoes.

## 3. Dados
- Fonte: CSVs publicos de voos, companhias e aeroportos (versionados com DVC).
- Local: data/raw (voos, airlines, airports).
- Observacao: dados de 2015, uso pre-voo (sem variaveis em tempo real).

## 4. Pipeline de dados
- Limpeza e filtros: remocao de cancelados e divertidos.
- Tratamento de variaveis categoricas: normalizacao e valores MISSING.
- Engenharia de atributos: tempo, rota, sazonalidade, congestao, estatisticas rolling.
- Split temporal: treino (meses 1-8), validacao (9-10), teste (11-12).

## 5. Modelo
- Algoritmo: CatBoostClassifier.
- Calibracao: Platt (LogisticRegression).
- Threshold otimizado para precision >= 0.66.
- Artefatos: catboost_model.cbm, platt_calibrator.joblib, best_threshold.txt, metadata.json, metrics.json, stats por aeroporto/companhia.

## 6. Serving / API
- FastAPI com endpoint /query.
- Entrada validada por InputGuardrail.
- Saida sanitizada por OutputGuardrail.
- Observabilidade via Prometheus.

## 7. Agente LLM
- Agente ReAct usando LLM configurado em configs/agent_config.yaml.
- Interpreta perguntas em linguagem natural e aciona tools.

## 8. Tools
- predict_flight_delay: predicao de atraso.
- get_airport_delay_stats: estatisticas por aeroporto.
- get_airline_delay_stats: estatisticas por companhia.
- search_flight_knowledge: RAG para recuperar contexto do knowledge base.

## 9. RAG
- Base: arquivos markdown em data/knowledge_base e docs.
- Embeddings: all-MiniLM-L6-v2.
- Vetor: FAISS.
- Chunking: 512 tokens, overlap 50.

## 10. Avaliacao
- Golden set com >= 20 pares.
- RAGAS (4 metricas): faithfulness, answer_relevancy, context_precision, context_recall.
- LLM-as-judge com 3 criterios.

## 11. Observabilidade
- Prometheus + Grafana para metricas de API.
- Drift detection com Evidently.
- MLflow para tracking de experimentos e artefatos.

## 12. Seguranca e governanca
- Guardrails de input: bloqueio de prompt injection e limite de tamanho.
- Guardrails de output: redacao de PII (email, telefone, CPF, cartao).
- OWASP mapping e red team report documentados.
- Model Registry com approval gate:
  - MLFLOW_MODEL_NAME define o nome do modelo.
  - MLFLOW_APPROVE controla promocao para Staging.

## 13. CI/CD
- GitHub Actions: lint, type check, security, tests.
- Build de imagens (train e serve).
- Deploy staging com smoke test /health.

## 14. Como executar (resumo)
- Treino: make docker-train ou make train
- Execução: make docker-serve ou make serve

## 15. Variaveis de ambiente
- MLFLOW_TRACKING_URI
- MLFLOW_MODEL_NAME
- MLFLOW_APPROVE
- GITHUB_TOKEN ou OPENAI_API_KEY

## 16. Estrutura principal do projeto
- src/features: feature engineering
- src/models: treino e predicao
- src/serving: API
- src/agent: agente, tools e RAG
- src/monitoring: drift e metricas
- src/security: guardrails e PII
- evaluation: ragas e llm judge
- docs: documentacao de governanca

## 17. Evidencias e links internos
- docs/MODEL_CARD.md
- docs/SYSTEM_CARD.md
- docs/RED_TEAM_REPORT.md
- docs/OWASP_MAPPING.md
- docs/LGPD_PLAN.md
- docs/benchmark.md

## 18. Observacoes finais
Este documento e um resumo executivo. Para detalhes tecnicos, consultar os arquivos citados na secao 17.
