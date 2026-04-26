# Benchmark — Configurações de LLM para o Agente AAPL

**Datathon FIAP Fase 05 | Grupo 29**

## Metodologia

3 perguntas de referência foram testadas contra cada configuração do agente:

1. "What is the AAPL price prediction for the next 5 days?"
2. "What is the risk of a position with 200 AAPL shares?"
3. "Do the current technical indicators suggest buying or selling?"

Métricas avaliadas:
- **Latência (s):** tempo total de resposta incluindo chamadas a tools
- **Qualidade (1–5):** avaliação manual — clareza, citação dos dados, utilidade
- **Tools usadas corretamente:** se o agente invocou a tool adequada

Ambiente: GitHub Models API (Azure), acesso via `GITHUB_TOKEN` (gratuito com GitHub Copilot). Inferência remota, sem GPU local necessária.

---

## Configuração A — gpt-4o-mini, temperatura 0.0, sem RAG

| Parâmetro | Valor |
|---|---|
| Modelo | gpt-4o-mini |
| Provider | GitHub Models (Azure inference endpoint) |
| Temperatura | 0.0 |
| Max tokens | 2048 |
| RAG | Desativado |
| Framework | LangChain + LangGraph `create_agent` |

| Pergunta | Latência (s) | Qualidade (1-5) | Tool correta |
|---|---|---|---|
| Previsão D+5 | ~3.1 | 4 | ✓ predict_price_delta |
| Risco de posição | ~2.8 | 4 | ✓ calculate_position_risk |
| Indicadores técnicos | ~2.6 | 4 | ✓ get_technical_indicators |

**Observações:** Respostas claras e bem estruturadas. Cita os dados retornados pelas tools. Sem RAG, não contextualiza regras de risco além do que a tool retorna diretamente.

---

## Configuração B — gpt-4o-mini, temperatura 0.0, com RAG

| Parâmetro | Valor |
|---|---|
| Modelo | gpt-4o-mini |
| Provider | GitHub Models (Azure inference endpoint) |
| Temperatura | 0.0 |
| Max tokens | 2048 |
| RAG | Ativo — FAISS + all-MiniLM-L6-v2 (local) |
| Knowledge base | `data/knowledge_base/` |
| Framework | LangChain + LangGraph `create_agent` |

| Pergunta | Latência (s) | Qualidade (1-5) | Tool correta |
|---|---|---|---|
| Previsão D+5 | ~3.8 | 5 | ✓ predict_price_delta + RAG |
| Risco de posição | ~3.4 | 5 | ✓ calculate_position_risk + RAG |
| Indicadores técnicos | ~3.2 | 5 | ✓ get_technical_indicators + RAG |

**Observações:** Melhor configuração geral. O RAG enriquece as respostas com contexto de interpretação de indicadores e regras de gestão de risco da knowledge base. Custo adicional de embeddings zero (sentence-transformers local). **Configuração padrão do projeto.**

---

## Configuração C — gpt-4o-mini, temperatura 0.3, com RAG (criatividade moderada)

| Parâmetro | Valor |
|---|---|
| Modelo | gpt-4o-mini |
| Provider | GitHub Models (Azure inference endpoint) |
| Temperatura | 0.3 |
| Max tokens | 2048 |
| RAG | Ativo — FAISS + all-MiniLM-L6-v2 (local) |
| Framework | LangChain + LangGraph `create_agent` |

| Pergunta | Latência (s) | Qualidade (1-5) | Tool correta |
|---|---|---|---|
| Previsão D+5 | ~3.9 | 4 | ✓ predict_price_delta + RAG |
| Risco de posição | ~3.5 | 4 | ✓ calculate_position_risk + RAG |
| Indicadores técnicos | ~3.3 | 4 | ✓ get_technical_indicators + RAG |

**Observações:** Temperatura mais alta gera respostas ligeiramente mais elaboradas, mas introduz variação entre execuções — indesejável em contexto financeiro. Não recomendado para produção.

---

## Resumo Comparativo

| Config | Modelo | Temperatura | RAG | Latência média | Qualidade média | Recomendação |
|---|---|---|---|---|---|---|
| A | gpt-4o-mini | 0.0 | Não | ~2.8s | 4.0/5 | Fallback sem RAG |
| B | gpt-4o-mini | 0.0 | Sim | ~3.5s | 5.0/5 | **Recomendado** |
| C | gpt-4o-mini | 0.3 | Sim | ~3.6s | 4.0/5 | Não recomendado |

**Configuração escolhida:** B — `gpt-4o-mini` com RAG e temperatura 0.0.

---

## Sobre o Modelo e Custo

O projeto utiliza **GitHub Models** como provider LLM:
- Endpoint: `https://models.inference.ai.azure.com`
- Autenticação: `GITHUB_TOKEN` (Personal Access Token)
- Custo: **gratuito** para usuários com GitHub Copilot
- Modelo: `gpt-4o-mini` — versão otimizada de custo do GPT-4o (OpenAI)
- Sem necessidade de GPU local ou infraestrutura própria de serving

Vantagens em relação a Ollama local:
- Latência menor (~3s vs ~6s) sem hardware dedicado
- Sem limite de VRAM (modelos locais Q4 comprometem qualidade)
- Gratuito via Copilot — sem impacto de custo no projeto

## Stack de Embeddings

- Modelo: `all-MiniLM-L6-v2` (sentence-transformers, 90 MB, 100% local)
- Vector store: FAISS (in-memory, sem infraestrutura adicional)
- Chunks: 512 tokens, overlap 50 tokens
- K resultados por query: 4
