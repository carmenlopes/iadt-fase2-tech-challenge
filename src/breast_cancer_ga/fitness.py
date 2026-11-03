"""Fitness por validacao cruzada estratificada exclusivamente no treino."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from .data import MALIGNANT_LABEL
from .models import Individual, ModelName, build_pipeline, validate_individual


class CandidateEvaluationError(RuntimeError):
    """Falha explicita de treinamento/avaliacao de um individuo."""


@dataclass(frozen=True)
class FitnessResult:
    fitness: float
    recall_mean: float
    recall_std: float
    precision_mean: float
    precision_std: float
    f1_mean: float
    f1_std: float
    false_negatives_mean: float
    false_negatives_total: int
    folds: list[dict[str, Any]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_individual(
    model: ModelName,
    individual: Individual,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = 5,
    seed: int = 42,
) -> FitnessResult:
    """Calcula recall maligno medio; nunca recebe nem acessa o conjunto de teste."""
    validate_individual(model, individual)
    class_counts = y_train.value_counts()
    if MALIGNANT_LABEL not in class_counts or int(class_counts.min()) < folds:
        raise ValueError("Cada classe do treino precisa ter ao menos 'folds' amostras.")

    pipeline = build_pipeline(model, individual, seed=seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_rows: list[dict[str, Any]] = []
    warning_messages: list[str] = []

    for fold_number, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        estimator = clone(pipeline)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
                prediction = estimator.predict(X_train.iloc[val_idx])
            convergence = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
            if convergence:
                details = "; ".join(str(w.message) for w in convergence)
                raise CandidateEvaluationError(
                    f"Convergencia falhou no fold {fold_number}: {details}"
                )
            warning_messages.extend(
                f"fold {fold_number}: {w.category.__name__}: {w.message}" for w in caught
            )
        except CandidateEvaluationError:
            raise
        except Exception as exc:
            raise CandidateEvaluationError(
                f"Falha no fold {fold_number}: {type(exc).__name__}: {exc}"
            ) from exc

        truth = y_train.iloc[val_idx]
        tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
        fold_rows.append(
            {
                "fold": fold_number,
                "recall": float(
                    recall_score(truth, prediction, pos_label=MALIGNANT_LABEL, zero_division=0)
                ),
                "precision": float(
                    precision_score(
                        truth, prediction, pos_label=MALIGNANT_LABEL, zero_division=0
                    )
                ),
                "f1": float(
                    f1_score(truth, prediction, pos_label=MALIGNANT_LABEL, zero_division=0)
                ),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "true_positives": int(tp),
            }
        )

    recalls = np.array([row["recall"] for row in fold_rows])
    precisions = np.array([row["precision"] for row in fold_rows])
    f1_values = np.array([row["f1"] for row in fold_rows])
    false_negatives = np.array([row["false_negatives"] for row in fold_rows])
    recall_mean = float(recalls.mean())
    return FitnessResult(
        fitness=recall_mean,
        recall_mean=recall_mean,
        recall_std=float(recalls.std(ddof=0)),
        precision_mean=float(precisions.mean()),
        precision_std=float(precisions.std(ddof=0)),
        f1_mean=float(f1_values.mean()),
        f1_std=float(f1_values.std(ddof=0)),
        false_negatives_mean=float(false_negatives.mean()),
        false_negatives_total=int(false_negatives.sum()),
        folds=fold_rows,
        warnings=warning_messages,
    )

