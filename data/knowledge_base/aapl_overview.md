# Apple Inc. (AAPL) — Visão Geral para Análise Financeira

## Perfil da Empresa
Apple Inc. é uma empresa de tecnologia americana que projeta, fabrica e comercializa
smartphones, computadores pessoais, tablets, wearables e acessórios, além de serviços digitais.
Ticker: AAPL (NASDAQ). Fundada em 1976, com sede em Cupertino, CA.

## Dados de Mercado (janela de treino do modelo: 2018–2026)
- Preço médio histórico no período: ~$150–$180 USD
- Volatilidade anualizada histórica: ~28%
- Capitalização de mercado: >$3 trilhões USD (2024)
- Volume médio diário: ~60 milhões de ações

## Indicadores Técnicos — Interpretação

### RSI-14 (Índice de Força Relativa)
- RSI > 70: sobrecompra — possível reversão de queda
- RSI < 30: sobrevenda — possível reversão de alta
- RSI entre 40–60: zona neutra, tendência indefinida

### MACD (Moving Average Convergence Divergence)
- MACD > Signal: momentum de alta (sinal bullish)
- MACD < Signal: momentum de baixa (sinal bearish)
- Cruzamento MACD-Signal: ponto de entrada/saída frequente

### Médias Móveis — SMA-7 e SMA-21
- SMA-7 > SMA-21: tendência de curto prazo positiva (golden cross)
- SMA-7 < SMA-21: tendência de curto prazo negativa (death cross)

### Volatilidade (vol_21)
- Volatilidade diária > 0.025: período de risco elevado
- Volatilidade diária < 0.010: mercado calmo, menor incerteza

## Modelo de Previsão LSTM — D+5
O modelo LSTM treinado prevê o delta de preço em USD para D+5:
- Fórmula: y = close(t+5) - close(t)
- MAE médio: $7.36 (erro esperado por ação em D+5)
- RMSE: $9.86
- Acurácia direcional: 52.5% (acima do acaso de 50%)
- Lookback: 60 dias de histórico como input
- Horizonte adequado: alocação de médio prazo, não trading de alta frequência

## Eventos Macroeconômicos que Afetam AAPL (2018–2026)
- 2020 Mar: COVID-19 — queda de ~30% seguida de rally acelerado
- 2022 Jan–Out: Ciclo de alta de juros Fed — queda de ~25%
- 2023–2024: Rally de IA — alta de ~50% impulsionada por interesse em tech
- 2025: Incerteza tarifária EUA-China e tensões geopolíticas

## Fatores de Risco Específicos da AAPL
- Dependência do iPhone (~50% da receita total)
- Concentração em fornecedores asiáticos (risco de cadeia de suprimento)
- Regulação antimonopólio em App Store (EUA e Europa)
- Concorrência crescente na China (Huawei, Xiaomi)
- Risco cambial: receitas globais expostas a variações do dólar
