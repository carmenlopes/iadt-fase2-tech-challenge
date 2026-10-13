"""Espacos de genes, validacao e construcao dos pipelines."""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Literal

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ModelName = Literal["logistic_regression", "decision_tree"]
Individual = dict[str, Any]

LOGISTIC_SOLVER_PENALTIES: dict[str, tuple[str, ...]] = {
    "lbfgs": ("l2",),
    "liblinear": ("l1", "l2"),
    "saga": ("l1", "l2"),
}
TREE_CRITERIA = ("gini", "entropy", "log_loss")


def sample_individual(model: ModelName, rng: random.Random) -> Individual:
    if model == "logistic_regression":
        solver = rng.choice(tuple(LOGISTIC_SOLVER_PENALTIES))
        return {
            "C": 10 ** rng.uniform(-4.0, 4.0),
            "solver": solver,
            "penalty": rng.choice(LOGISTIC_SOLVER_PENALTIES[solver]),
        }
    if model == "decision_tree":
        return {
            "max_depth": rng.choice([None, *range(1, 21)]),
            "min_samples_split": rng.randint(2, 30),
            "min_samples_leaf": rng.randint(1, 15),
            "criterion": rng.choice(TREE_CRITERIA),
        }
    raise ValueError(f"Modelo desconhecido: {model}")


def validate_individual(model: ModelName, individual: Individual) -> None:
    if model == "logistic_regression":
        expected = {"C", "solver", "penalty"}
        if set(individual) != expected:
            raise ValueError(f"Genes logisticos devem ser {sorted(expected)}")
        c_value = individual["C"]
        if isinstance(c_value, bool) or not isinstance(c_value, (int, float)):
            raise ValueError("C deve ser numerico.")
        if not math.isfinite(float(c_value)) or not 1e-4 <= float(c_value) <= 1e4:
            raise ValueError("C deve estar no intervalo [1e-4, 1e4].")
        solver = individual["solver"]
        if solver not in LOGISTIC_SOLVER_PENALTIES:
            raise ValueError(f"Solver invalido: {solver}")
        if individual["penalty"] not in LOGISTIC_SOLVER_PENALTIES[solver]:
            raise ValueError(
                f"Combinacao invalida: solver={solver}, penalty={individual['penalty']}"
            )
        return
    if model == "decision_tree":
        expected = {"max_depth", "min_samples_split", "min_samples_leaf", "criterion"}
        if set(individual) != expected:
            raise ValueError(f"Genes da arvore devem ser {sorted(expected)}")
        depth = individual["max_depth"]
        if depth is not None and (type(depth) is not int or not 1 <= depth <= 20):
            raise ValueError("max_depth deve ser None ou inteiro entre 1 e 20.")
        split = individual["min_samples_split"]
        leaf = individual["min_samples_leaf"]
        if type(split) is not int or not 2 <= split <= 30:
            raise ValueError("min_samples_split deve ser inteiro entre 2 e 30.")
        if type(leaf) is not int or not 1 <= leaf <= 15:
            raise ValueError("min_samples_leaf deve ser inteiro entre 1 e 15.")
        if individual["criterion"] not in TREE_CRITERIA:
            raise ValueError("criterion invalido.")
        return
    raise ValueError(f"Modelo desconhecido: {model}")


def repair_individual(model: ModelName, individual: Individual, rng: random.Random) -> Individual:
    """Retorna copia valida; em especial repara solver/penalty apos crossover."""
    repaired = copy.deepcopy(individual)
    if model == "logistic_regression":
        solver = repaired.get("solver")
        if solver not in LOGISTIC_SOLVER_PENALTIES:
            solver = rng.choice(tuple(LOGISTIC_SOLVER_PENALTIES))
            repaired["solver"] = solver
        if repaired.get("penalty") not in LOGISTIC_SOLVER_PENALTIES[solver]:
            repaired["penalty"] = rng.choice(LOGISTIC_SOLVER_PENALTIES[solver])
        c_value = repaired.get("C")
        if not isinstance(c_value, (int, float)) or isinstance(c_value, bool):
            repaired["C"] = 10 ** rng.uniform(-4.0, 4.0)
        else:
            repaired["C"] = min(1e4, max(1e-4, float(c_value)))
    elif model == "decision_tree":
        valid_depths = [None, *range(1, 21)]
        if repaired.get("max_depth") not in valid_depths:
            repaired["max_depth"] = rng.choice(valid_depths)
        if type(repaired.get("min_samples_split")) is not int:
            repaired["min_samples_split"] = rng.randint(2, 30)
        repaired["min_samples_split"] = min(
            30, max(2, repaired["min_samples_split"])
        )
        if type(repaired.get("min_samples_leaf")) is not int:
            repaired["min_samples_leaf"] = rng.randint(1, 15)
        repaired["min_samples_leaf"] = min(15, max(1, repaired["min_samples_leaf"]))
        if repaired.get("criterion") not in TREE_CRITERIA:
            repaired["criterion"] = rng.choice(TREE_CRITERIA)
    else:
        raise ValueError(f"Modelo desconhecido: {model}")
    validate_individual(model, repaired)
    return repaired


def mutate_individual(
    model: ModelName, individual: Individual, mutation_rate: float, rng: random.Random
) -> Individual:
    """Muta uma copia, preservando o pai e os dominios dos genes."""
    child = copy.deepcopy(individual)
    if model == "logistic_regression":
        if rng.random() < mutation_rate:
            child["C"] = 10 ** rng.uniform(-4.0, 4.0)
        if rng.random() < mutation_rate:
            child["solver"] = rng.choice(tuple(LOGISTIC_SOLVER_PENALTIES))
        if rng.random() < mutation_rate:
            child["penalty"] = rng.choice(
                LOGISTIC_SOLVER_PENALTIES[child["solver"]]
            )
    elif model == "decision_tree":
        if rng.random() < mutation_rate:
            child["max_depth"] = rng.choice([None, *range(1, 21)])
        if rng.random() < mutation_rate:
            child["min_samples_split"] = rng.randint(2, 30)
        if rng.random() < mutation_rate:
            child["min_samples_leaf"] = rng.randint(1, 15)
        if rng.random() < mutation_rate:
            child["criterion"] = rng.choice(TREE_CRITERIA)
    else:
        raise ValueError(f"Modelo desconhecido: {model}")
    return repair_individual(model, child, rng)


def build_pipeline(model: ModelName, individual: Individual, *, seed: int) -> Pipeline:
    validate_individual(model, individual)
    if model == "logistic_regression":
        estimator = LogisticRegression(
            C=float(individual["C"]),
            solver=individual["solver"],
            penalty=individual["penalty"],
            max_iter=2_000,
            random_state=seed,
        )
        return Pipeline([("preprocess", StandardScaler()), ("model", estimator)])
    estimator = DecisionTreeClassifier(random_state=seed, **individual)
    return Pipeline([("preprocess", "passthrough"), ("model", estimator)])

