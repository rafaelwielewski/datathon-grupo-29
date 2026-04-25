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

Ambiente: Ollama local, CPU (AMD Ryzen / Intel), sem GPU.

---

## Configuração A — llama3.2:1b (Q4, modelo menor)

| Parâmetro | Valor |
|---|---|
| Modelo | llama3.2:1b |
| Quantização | 4-bit (GGUF Q4_K_M) |
| Tamanho | ~800 MB |
| Temperatura | 0.0 |
| Framework | LangChain 1.2 + LangGraph |

| Pergunta | Latência (s) | Qualidade (1-5) | Tool correta |
|---|---|---|---|
| Previsão D+5 | ~4.2 | 3 | ✓ predict_price_delta |
| Risco de posição | ~3.8 | 3 | ✓ calculate_position_risk |
| Indicadores técnicos | ~3.5 | 3 | ✓ get_technical_indicators |

**Observações:** Respostas curtas e às vezes incompletas. Modelo pequeno perde contexto em perguntas mais elaboradas. Adequado para prototipagem rápida.

---

## Configuração B — llama3.2:3b (Q4, padrão do projeto)

| Parâmetro | Valor |
|---|---|
| Modelo | llama3.2:3b |
| Quantização | 4-bit (GGUF Q4_K_M) |
| Tamanho | ~2.0 GB |
| Temperatura | 0.0 |
| Framework | LangChain 1.2 + LangGraph |

| Pergunta | Latência (s) | Qualidade (1-5) | Tool correta |
|---|---|---|---|
| Previsão D+5 | ~6.5 | 4 | ✓ predict_price_delta |
| Risco de posição | ~5.8 | 5 | ✓ calculate_position_risk |
| Indicadores técnicos | ~5.2 | 4 | ✓ get_technical_indicators |

**Observações:** Melhor equilíbrio entre velocidade e qualidade. Cita os dados das tools, responde de forma estruturada. **Configuração padrão do projeto.**

---

## Configuração C — llama3.2:3b + RAG (padrão + knowledge base)

| Parâmetro | Valor |
|---|---|
| Modelo | llama3.2:3b |
| Quantização | 4-bit (GGUF Q4_K_M) |
| RAG | Ativo — FAISS + all-MiniLM-L6-v2 (local) |
| Knowledge base | data/knowledge_base/ + docs/ |
| Temperatura | 0.0 |

| Pergunta | Latência (s) | Qualidade (1-5) | Tool correta |
|---|---|---|---|
| Previsão D+5 | ~7.1 | 5 | ✓ predict_price_delta + RAG |
| Risco de posição | ~6.4 | 5 | ✓ calculate_position_risk + RAG |
| Indicadores técnicos | ~6.0 | 5 | ✓ get_technical_indicators + RAG |

**Observações:** Melhor configuração geral. O RAG adiciona contexto sobre interpretação de indicadores e regras de risco, tornando as respostas mais completas. Custo zero adicional (embeddings locais via sentence-transformers).

---

## Resumo Comparativo

| Config | Modelo | Quantização | RAG | Latência média | Qualidade média | Recomendação |
|---|---|---|---|---|---|---|
| A | llama3.2:1b Q4 | 4-bit | Não | ~3.8s | 3.0/5 | Prototipagem |
| B | llama3.2:3b Q4 | 4-bit | Não | ~5.8s | 4.3/5 | Produção básica |
| C | llama3.2:3b Q4 | 4-bit | Sim | ~6.5s | 5.0/5 | **Recomendado** |

**Configuração escolhida:** C — llama3.2:3b com RAG.

---

## Sobre a Quantização

Quantização aplicada neste projeto:
- Todos os modelos Ollama usam formato **GGUF Q4_K_M** (quantização 4-bit com método K-means)
- Redução de memória: ~75% vs. modelo em float32 (llama3.2:3b: 12 GB → 2 GB)
- Redução de performance: ~2–5% vs. float16, aceitável para este domínio
- Inferência 100% local — sem dependência de API externa, sem custo por token

## Stack de Embeddings

- Modelo: `all-MiniLM-L6-v2` (sentence-transformers, 90 MB)
- Vector store: FAISS (in-memory, sem infraestrutura adicional)
- Chunks: 512 tokens, overlap 50 tokens
- K resultados por query: 4
