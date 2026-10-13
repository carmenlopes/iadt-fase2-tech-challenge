from __future__ import annotations

import copy
import random

import pytest

from breast_cancer_ga.models import (
    LOGISTIC_SOLVER_PENALTIES,
    mutate_individual,
    repair_individual,
    sample_individual,
    validate_individual,
)


@pytest.mark.parametrize("model", ["logistic_regression", "decision_tree"])
def test_generated_individuals_are_valid(model: str) -> None:
    rng = random.Random(42)
    for _ in range(100):
        validate_individual(model, sample_individual(model, rng))


def test_all_logistic_solver_penalty_combinations_are_compatible() -> None:
    for solver, penalties in LOGISTIC_SOLVER_PENALTIES.items():
        for penalty in penalties:
            validate_individual(
                "logistic_regression", {"C": 1.0, "solver": solver, "penalty": penalty}
            )
    with pytest.raises(ValueError):
        validate_individual(
            "logistic_regression", {"C": 1.0, "solver": "lbfgs", "penalty": "l1"}
        )


def test_mutation_preserves_parent_and_validity() -> None:
    parent = {"C": 1.0, "solver": "lbfgs", "penalty": "l2"}
    snapshot = copy.deepcopy(parent)
    child = mutate_individual("logistic_regression", parent, 1.0, random.Random(7))
    assert parent == snapshot
    assert child is not parent
    validate_individual("logistic_regression", child)


def test_repair_fixes_invalid_solver_penalty_pair() -> None:
    repaired = repair_individual(
        "logistic_regression",
        {"C": 1.0, "solver": "lbfgs", "penalty": "l1"},
        random.Random(1),
    )
    validate_individual("logistic_regression", repaired)

