# System Card — Flight Delay Prediction Assistant

**Datathon FIAP Fase 05 | Grupo 29**

## 1. Visão geral
- Sistema: agente ReAct para predição de atrasos em voos domésticos dos EUA.
- Objetivo: suporte a decisão de viagem, não garantia operacional.
- Usuários: avaliadores e demonstração acadêmica.

## 2. Componentes
- API: FastAPI em `src/serving/app.py`.
- Agente: ReAct + 4 tools (predict_flight_delay, get_airport_delay_stats, get_airline_delay_stats, search_flight_knowledge).
- Modelo: CatBoost + calibração Platt, artefatos em `data/processed/artifacts/`.
- RAG: FAISS + embeddings locais (knowledge base em `data/knowledge_base` e `docs`).
- Monitoramento: Prometheus + Evidently (drift).

## 3. Dados
- Fonte: dataset público de voos domésticos EUA 2015 (~5M registros).
- PII: não coletado; redação aplicada em output via OutputGuardrail.

## 4. Guardrails e segurança
- Input guardrail: detecção de prompt injection e limite de tamanho (1000 chars).
- Output guardrail: redação de PII (e-mail, telefone, CPF, cartão).
- Tools limitadas a leitura de artefatos locais e inferência — sem escrita ou execução remota.

## 5. Riscos conhecidos
- Predições baseadas em dados de 2015 — padrões operacionais atuais podem diferir.
- Prompt injection pode evoluir; guardrails precisam revisão periódica.
- Placeholders de features (ORIGIN_STATE, TAIL_NUMBER) reduzem precisão para voos específicos.

## 6. Monitoramento
- Métricas de API: latência, throughput, erros (Prometheus + Grafana).
- Drift: share de features com drift e thresholds via Evidently.

## 7. Uso pretendido
- Suporte informativo a decisões de viagem aérea doméstica nos EUA.
- Não usar para decisões operacionais críticas ou planejamento de frota.

## 8. Contato
- Grupo 29 (FIAP) — responsáveis pela demonstração.
