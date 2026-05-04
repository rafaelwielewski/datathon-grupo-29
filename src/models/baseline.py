from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    result = {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        result['auc'] = float(roc_auc_score(y_true, y_proba))
    return result


class MajorityClassBaseline:
    def __init__(self) -> None:
        self._majority_class = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MajorityClassBaseline':
        self._majority_class = int(np.argmax(np.bincount(y.astype(int))))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._majority_class, dtype=int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return classification_metrics(y, self.predict(X))


class PriorRateBaseline:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._prior = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PriorRateBaseline':
        self._prior = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred_class = 1 if self._prior >= self.threshold else 0
        return np.full(len(X), pred_class, dtype=int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return classification_metrics(y, self.predict(X))


class LogisticRegressionBaseline:
    def __init__(self) -> None:
        self.model = LogisticRegression(random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionBaseline':
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return classification_metrics(y, self.predict(X), self.predict_proba(X))
