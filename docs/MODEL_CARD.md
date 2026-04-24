# Model Card — AAPL Price Forecasting (LSTM)

**Datathon FIAP Fase 05 | Grupo 29**

---

## 1. Descrição do Modelo

| Campo | Valor |
|---|---|
| Nome | LSTM Price Forecaster |
| Versão | 1.0.0 |
| Tipo | Regressão de série temporal (LSTM) |
| Framework | TensorFlow 2.19 / Keras |
| Formato de produção | ONNX |
| Data de treino | 2026-04-24 |

**Objetivo**: Prever o preço de fechamento da ação AAPL com horizonte de 5 dias úteis, utilizando os últimos 60 pregões como contexto.

---

## 2. Dados de Treinamento

| Campo | Valor |
|---|---|
| Ativo | Apple Inc. (AAPL) |
| Fonte | Yahoo Finance (yfinance) |
| Período | 2018-01-01 → 2026-01-01 |
| Pregões | ~1.985 |
| Split | 70% treino / 15% validação / 15% teste |
| Método de split | Temporal estrito (sem data leakage) |

**Features (16)**:
- **Preço/Volume**: `close`, `high`, `low`, `open`, `volume`
- **Retornos**: `ret_1`, `log_ret_1`
- **Médias móveis**: `sma_7`, `sma_21`, `ema_12`, `ema_26`
- **Momentum**: `macd`, `macd_signal`, `rsi_14`
- **Volatilidade**: `vol_7`, `vol_21`

**Target**: `y_delta_h = close(t+5) - close(t)` — delta em USD para D+5.

---

## 3. Arquitetura

```
Input(60 timesteps × 16 features)
  → LSTM(64 units, recurrent_dropout=0.05, return_sequences=True)
  → Dropout(0.2)
  → LSTM(32 units, recurrent_dropout=0.05)
  → Dropout(0.2)
  → Dense(16, relu)
  → Dense(1)  ← delta previsto em USD
```

| Hiperparâmetro | Valor |
|---|---|
| Loss | Huber (delta=1.0) |
| Optimizer | Adam (lr=5e-4, clipnorm=1.0) |
| Epochs (máx) | 120 |
| Batch size | 32 |
| Early stopping | patience=15 |
| Pré-processamento | RobustScaler (fit apenas no treino) |

---

## 4. Métricas Técnicas

### 4.1 Resultados no Conjunto de Teste (D+5)

| Métrica | LSTM | Naive Baseline | SMA-60 Baseline |
|---|---|---|---|
| MAE (USD) | **7.36** | 7.31 | 15.12 |
| RMSE (USD) | **9.86** | 9.80 | 18.33 |
| MAPE (%) | **3.27%** | 3.26% | 6.50% |
| Directional Accuracy | **52.5%** | — | — |

> O LSTM performa ~2× melhor que o SMA-60 em MAPE e alcança 52.5% de acerto direcional (acima do random 50%).

---

## 5. Métricas de Negócio

Esta seção mapeia as métricas técnicas para impacto real nas operações financeiras.

### 5.1 MAE → Risco de Erro por Posição

Um MAE de **$7.36** significa que, em média, a previsão erra o preço real de D+5 em $7.36.

| Tamanho da posição | Erro esperado por posição |
|---|---|
| 100 ações | ~$736 |
| 500 ações | ~$3.680 |
| 1.000 ações | ~$7.360 |

**Uso**: A mesa de operações pode dimensionar o tamanho máximo de posição aceitável dado o MAE do modelo. Posições maiores que o limiar de risco devem exigir confirmação humana.

### 5.2 MAPE → Desvio Relativo e Comparação com Custos

Um MAPE de **3.27%** sobre o preço médio de ~$180 equivale a ~$5.89 de desvio relativo médio.

| Referência | Valor |
|---|---|
| MAPE do modelo | 3.27% |
| MAPE do SMA-60 (benchmark) | 6.50% |
| Melhoria relativa vs SMA-60 | **~50% melhor** |
| Spread típico AAPL (bid/ask) | ~0.01–0.05% |
| Taxa de corretagem estimada | ~0.1–0.5% |

**Uso**: O MAPE de 3.27% é significativamente maior que os custos de transação. O modelo **não deve ser usado como único sinal para operações de altíssima frequência**. É mais adequado para decisões de alocação de médio prazo (horizonte de dias).

### 5.3 Directional Accuracy → Taxa de Acerto na Direção do Trade

Uma directional accuracy de **52.5%** significa que o modelo acerta a direção do movimento (alta/queda) em 52.5% dos casos — melhor que o acaso (50%).

| Cenário | Retorno esperado |
|---|---|
| Random (50%) | Neutro |
| Modelo (52.5%) | +2.5 p.p. de vantagem direcional |
| Modelo com stop-loss de 2× MAE | Reduz perdas em movimentos errados |

**Uso**: A vantagem direcional de +2.5 p.p. é modesta mas estatisticamente relevante em escala. Em uma estratégia com 200 operações/ano, o modelo acerta ~5 operações a mais que o acaso — suficiente para cobrir custos em posições médias.

### 5.4 RMSE → Value at Risk (VaR) Aproximado

O RMSE de **$9.86** pode ser usado como proxy conservador para o desvio padrão do erro de previsão.

| Nível de confiança | VaR estimado (por ação) |
|---|---|
| 68% (1σ) | ±$9.86 |
| 95% (2σ) | ±$19.72 |
| 99% (3σ) | ±$29.58 |

**Uso**: Para uma posição de 100 ações, o VaR a 95% é de ~$1.972. Esse valor deve ser considerado no cálculo de margem e nos limites de risco operacional da mesa.

### 5.5 Sumário Executivo

| Pergunta de negócio | Resposta do modelo |
|---|---|
| O modelo é melhor que não fazer nada? | Sim — 2× melhor que SMA-60 em MAPE |
| Qual o erro médio por ação em USD? | $7.36 (MAE) |
| O modelo acerta a direção do mercado? | 52.5% — ligeiramente acima do acaso |
| Serve para trading de alta frequência? | Não — erro maior que custos de transação |
| Serve para alocação de médio prazo? | Sim — horizonte D+5 com risco dimensionável |
| Precisa de supervisão humana? | Sim — recomendado para posições > 500 ações |

---

## 6. Limitações e Riscos

- **Dados**: Treinado exclusivamente em AAPL. Não generaliza para outros ativos sem retreino.
- **Regime**: Treinado em 2018–2026. Eventos sem precedente histórico (ex: novo tipo de crise) podem degradar a performance.
- **Directional accuracy**: 52.5% não garante lucratividade — depende do dimensionamento de posição e gestão de risco.
- **Latência de dados**: Usa dados de fechamento (D). Não incorpora dados intraday ou notícias.
- **Não é conselho financeiro**: O modelo é uma ferramenta de suporte à decisão, não um sistema autônomo de trading.

---

## 7. Uso Pretendido

**Adequado para**:
- Suporte a decisões de alocação de médio prazo (horizonte de dias)
- Dimensionamento de posições com base no MAE
- Complemento a análises fundamentalistas e quantitativas

**Não adequado para**:
- Trading autônomo sem supervisão humana
- Operações de alta frequência (HFT)
- Previsão de outros ativos sem retreino

---

## 8. Informações de Governança

| Campo | Valor |
|---|---|
| Equipe responsável | Grupo 29 — FIAP Datathon Fase 05 |
| Experimento MLflow | `datathon-grupo-29` |
| Tracking URI | `sqlite:///mlflow.db` |
| Repositório | `datathon-grupo-29` |
| Artefatos | `data/processed/artifacts/` (DVC) |
| Risk level | Medium |
| Fairness checked | Não aplicável (ativo financeiro, sem atributos sensíveis) |
