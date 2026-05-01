# System Card - AAPL Financial Assistant

## 1. Visao geral
- Sistema: agente ReAct para analise AAPL com LSTM D+5 e RAG.
- Objetivo: suporte a decisao, nao trading autonomo.
- Usuarios: avaliadores e demonstracao academica.

## 2. Componentes
- API: FastAPI em src/serving/app.py.
- Agente: ReAct + tools (indicadores, previsao, risco).
- RAG: knowledge base em data/knowledge_base e docs/.
- Monitoramento: Prometheus + Evidently (drift).

## 3. Dados
- Fonte principal: Yahoo Finance (dados publicos).
- KB: documentos internos do projeto.
- PII: nao coletado; redacao aplicada em output.

## 4. Guardrails e seguranca
- Input guardrail com deteccao de prompt injection e limite de tamanho.
- Output guardrail com redacao de PII (email, telefone, CPF, CNPJ, CEP, cartao).
- Tools limitadas, sem capacidade de escrita ou execucao remota.

## 5. Riscos conhecidos
- Erro de previsao pode induzir decisoes ruins.
- Dependencia de dados publicos (latencia/indisponibilidade).
- Prompt injection ainda pode evoluir; precisa revisao periodica.

## 6. Monitoramento
- Metricas de API: latencia, throughput, erros.
- Drift: share de features com drift e thresholds.

## 7. Uso pretendido
- Suporte a analise de medio prazo.
- Nao usar para HFT ou decisoes autonomas.

## 8. Contato
- Grupo 29 (FIAP) - responsaveis pela demonstracao.
