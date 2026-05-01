# LGPD Plan - Datathon Fase 05 (Grupo 29)

## 1. Escopo
- Caso: previsao de preco AAPL (dados financeiros publicos).
- Dados pessoais: nao coletamos dados pessoais de usuarios finais.
- Possivel PII: entradas livres do usuario podem conter PII; saidas do LLM podem reproduzir PII.

## 2. Base legal e finalidade
- Base legal: legitimo interesse para demonstracao academica e pesquisa aplicada.
- Finalidade: analise tecnica e previsao de preco; suporte a decisao.
- Minimizacao: apenas dados de mercado e textos publicos do KB.

## 3. Inventario de dados
| Categoria | Origem | Retencao | Sensibilidade | Controles |
|---|---|---|---|---|
| OHLCV AAPL | Yahoo Finance (yfinance) | 24 meses | Publico | Versionamento DVC, acesso local |
| Knowledge base | docs/ e data/knowledge_base | Enquanto durar o projeto | Baixa | Revisao manual, sem PII |
| Logs de API | FastAPI | 30 dias | Baixa | Rotacao de logs, sem payload completo |

## 4. Medidas tecnicas
- Input guardrail: bloqueio de prompt injection e limite de tamanho.
- Output guardrail: redacao de PII (email, telefone, CPF, CNPJ, CEP, cartao) via regex.
- Segredo: tokens em .env e nunca hardcoded.
- Observabilidade: logs e metricas sem PII.

## 5. Direitos do titular
- Caso haja PII em entrada, o sistema aplica redacao no output.
- Nao ha armazenamento persistente de PII; logs nao devem conter PII.

## 6. Retencao e descarte
- Logs: 30 dias, rotacionados.
- Artefatos de modelo: mantidos ate o fim do projeto.

## 7. Responsaveis
- Controlador/operador: Grupo 29 (FIAP).
- Contato: canal do time durante a avaliacao.
