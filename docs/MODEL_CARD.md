# Model Card — Flight Delay Prediction (CatBoost)

**Datathon FIAP Fase 05 | Grupo 29**

---

## 1. Descrição do Modelo

| Campo | Valor |
|---|---|
| Nome | CatBoost Flight Delay Classifier |
| Versão | 1.0.0 |
| Tipo | Classificação binária |
| Framework | CatBoost 1.2+ |
| Calibração | Platt (LogisticRegression) + prior-shift adjustment |
| Data de treino | 2026-05-01 |

**Objetivo**: Prever se um voo doméstico nos EUA chegará com 15+ minutos de atraso,
usando apenas informações disponíveis antes da decolagem (pré-voo).

---

## 2. Dados de Treinamento

| Campo | Valor |
|---|---|
| Dataset | US Domestic Flights 2015 (Bureau of Transportation Statistics) |
| Arquivo | flights.csv (5.8M linhas, 565 MB) |
| Split temporal | Treino: meses 1-8 \| Validação: 9-10 \| Teste: 11-12 |
| Taxa de atraso (treino) | ~18% |
| Variável alvo | `delayed` = 1 se ARRIVAL_DELAY >= 15 minutos |

---

## 3. Features

### Base (sem histórico operacional): 37 features
Inclui: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, ROUTE, DISTANCE, SCHEDULED_TIME,
sched_dep_hour/period, sched_arr_hour/period, is_weekend, distance_bucket,
doy_sin/cos, congestion logvol, rolling target encodings (7d, 30d).

### Operacionais (com histórico): +16 features
Inclui: tail cascade delays, origin departure delay history, weather/system/late-aircraft rates,
route delay history.

---

## 4. Pipeline de Inferência

```
Parâmetros do voo → build_inference_row → CatBoostClassifier.predict_proba
  → Platt calibration → prior-shift adjustment → threshold (0.607) → delayed: bool
```

Features históricas ausentes são preenchidas com `TRAIN_PRIOR = 0.18`.

---

## 5. Métricas (Test Set — Meses 11-12)

| Métrica | Valor |
|---|---|
| AUC-ROC | ~0.72 |
| Precision | >= 0.66 (por design) |
| Recall | ~0.51 |
| F1 | ~0.58 |
| Threshold | ~0.607 |

---

## 6. Limitações

- Treinado em dados de 2015 — padrões operacionais podem ter mudado.
- Não usa informações em tempo real (clima, NOTAM, crowding).
- Aeroportos/companhias fora do dataset de treino recebem codificação MISSING.
- Não suporta voos internacionais.

---

## 7. Informações de Governança

| Campo | Valor |
|---|---|
| Equipe responsável | Grupo 29 — FIAP Datathon Fase 05 |
| Experimento MLflow | `datathon-grupo-29` |
| Tracking URI | `sqlite:///mlflow.db` |
| Repositório | `datathon-grupo-29` |
| Artefatos | `data/processed/artifacts/` (DVC) |
| Risk level | Medium |
| Fairness checked | Não aplicável (predição operacional, sem atributos sensíveis) |
