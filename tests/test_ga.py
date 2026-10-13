from __future__ import annotations

import copy

import pandas as pd
import pytest

from breast_cancer_ga.fitness import FitnessResult
from breast_cancer_ga.ga import GAConfig, GeneticAlgorithm
from breast_cancer_ga.models import validate_individual


def fake_fitness(model, individual, X_train, y_train, *, folds, seed):
    if model == "logistic_regression":
        score = 1.0 / (1.0 + abs(float(individual["C"]) - 1.0))
    else:
        score = 1.0 / (1.0 + abs((individual["max_depth"] or 20) - 4))
    return FitnessResult(score, score, 0.0, score, 0.0, score, 0.0, 0.0, 0, [], [])


@pytest.mark.parametrize("model", ["logistic_regression", "decision_tree"])
def test_complete_run_population_size_validity_and_elitism(model: str) -> None:
    X = pd.DataFrame({"x": range(20)})
    y = pd.Series([0, 1] * 10)
    config = GAConfig(model=model, population_size=8, generations=3, elitism=2, folds=2)
    ga = GeneticAlgorithm(config, X, y, fitness_function=fake_fitness)
    initial = ga.initial_population()
    initial_snapshot = copy.deepcopy(initial)
    initial_best = max(fake_fitness(model, item, X, y, folds=2, seed=42).fitness for item in initial)
    result = ga.run(initial)
    assert initial == initial_snapshot
    assert len(result.final_population) == config.population_size
    assert len(result.history) == config.generations
    assert result.best.metrics is not None
    assert result.best.metrics.fitness >= initial_best
    assert all(
        result.history[index]["best_fitness_recall"]
        <= result.history[index + 1]["best_fitness_recall"]
        for index in range(len(result.history) - 1)
    )
    for individual in result.final_population:
        validate_individual(model, individual)


def test_crossover_does_not_modify_parents() -> None:
    X = pd.DataFrame({"x": range(10)})
    y = pd.Series([0, 1] * 5)
    ga = GeneticAlgorithm(
        GAConfig(model="decision_tree", population_size=4, generations=1, folds=2),
        X, y, fitness_function=fake_fitness,
    )
    a = {"max_depth": 2, "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "gini"}
    b = {"max_depth": 8, "min_samples_split": 10, "min_samples_leaf": 4, "criterion": "entropy"}
    before_a, before_b = copy.deepcopy(a), copy.deepcopy(b)
    child_a, child_b = ga.crossover(a, b)
    assert a == before_a and b == before_b
    validate_individual("decision_tree", child_a)
    validate_individual("decision_tree", child_b)

