from __future__ import annotations

from pathlib import Path

import numpy as np

from breast_cancer_ga.data import MALIGNANT_LABEL, make_original_split
from breast_cancer_ga.fitness import evaluate_individual

ROOT = Path(__file__).resolve().parents[1]


def test_malignant_class_and_original_split() -> None:
    split = make_original_split(ROOT / "dataset-breast-cancer" / "data.csv")
    assert MALIGNANT_LABEL == 1
    assert len(split.X_train) == 455
    assert len(split.X_test) == 114
    assert int(split.y_test.sum()) == 42


def test_false_negatives_match_each_fold_confusion_matrix() -> None:
    split = make_original_split(ROOT / "dataset-breast-cancer" / "data.csv")
    result = evaluate_individual(
        "decision_tree",
        {"max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "gini"},
        split.X_train,
        split.y_train,
        folds=3,
        seed=42,
    )
    assert result.false_negatives_total == sum(row["false_negatives"] for row in result.folds)
    assert all(row["false_negatives"] >= 0 for row in result.folds)


def test_fitness_api_has_no_test_set_parameter() -> None:
    assert "X_test" not in evaluate_individual.__annotations__
    assert "y_test" not in evaluate_individual.__annotations__

