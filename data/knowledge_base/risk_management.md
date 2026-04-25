# Guia de Gestão de Risco — Posições em AAPL

## Métricas do Modelo para Dimensionamento de Posição

O modelo LSTM de previsão D+5 possui as seguintes métricas de erro no conjunto de teste:
- MAE: $7.36 por ação
- RMSE: $9.86 por ação
- MAPE: 3.27%
- Directional Accuracy: 52.5%

## Regras de Dimensionamento por Tamanho de Posição

| Posição (ações) | Erro Esperado (MAE) | VaR 95% (2x RMSE) | Classificação |
|---|---|---|---|
| até 100 | até $736 | até $1.972 | BAIXO RISCO |
| 101 a 500 | $736 a $3.680 | $1.972 a $9.860 | MÉDIO RISCO |
| acima de 500 | acima de $3.680 | acima de $9.860 | ALTO RISCO — revisão humana obrigatória |

## Critérios de Entrada em Posição LONG

Entrar comprado quando:
- Modelo prevê delta positivo maior que $5 (acima do MAE)
- RSI-14 entre 40 e 65 (não sobrecomprado)
- MACD positivo (momentum de alta)
- Volume do dia acima da média de 7 dias

Evitar posição quando:
- RSI-14 acima de 70 (sobrecompra) ou abaixo de 30 (sobrevenda extrema)
- vol_21 acima de 0.025 (volatilidade elevada)
- Divulgação de resultados (earnings) em menos de 5 dias úteis

## Regras de Stop-Loss e Take-Profit

- Stop-loss: 2x MAE do modelo ($14.72 por ação)
- Take-profit: 3x MAE do modelo ($22.08 por ação)
- Trailing stop: ativar quando posição estiver 1.5x MAE no positivo

## Value at Risk (VaR) por Nível de Confiança

| Confiança | Sigma | VaR por Ação | VaR (100 ações) |
|---|---|---|---|
| 68% | 1 sigma | $9.86 | $986 |
| 95% | 2 sigma | $19.72 | $1.972 |
| 99% | 3 sigma | $29.58 | $2.958 |

## Limitações do Modelo

- Treinado apenas em AAPL: não generaliza para outros ativos sem retreino
- Não incorpora dados intraday, notícias ou sentimento de mercado
- Eventos sem precedente histórico como crises sistêmicas podem degradar a performance
- O modelo não é conselho financeiro: é ferramenta de suporte à decisão
- Supervisão humana recomendada para posições acima de 500 ações
