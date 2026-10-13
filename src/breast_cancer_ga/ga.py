"""Implementacao compartilhada do algoritmo genetico."""

from __future__ import annotations

import copy
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import pandas as pd

from .fitness import CandidateEvaluationError, FitnessResult, evaluate_individual
from .models import (
    Individual,
    ModelName,
    mutate_individual,
    repair_individual,
    sample_individual,
    validate_individual,
)


@dataclass(frozen=True)
class GAConfig:
    model: ModelName
    seed: int = 42
    population_size: int = 8
    generations: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elitism: int = 1
    folds: int = 5
    tournament_size: int = 3

    def validate(self) -> None:
        if self.model not in ("logistic_regression", "decision_tree"):
            raise ValueError("Modelo invalido.")
        if self.population_size < 2:
            raise ValueError("population_size deve ser >= 2.")
        if self.generations < 1:
            raise ValueError("generations deve ser >= 1.")
        if not 0 <= self.crossover_rate <= 1 or not 0 <= self.mutation_rate <= 1:
            raise ValueError("Taxas devem estar no intervalo [0, 1].")
        if not 1 <= self.elitism < self.population_size:
            raise ValueError("elitism deve estar entre 1 e population_size - 1.")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size invalido.")
        if self.folds < 2:
            raise ValueError("folds deve ser >= 2.")


@dataclass(frozen=True)
class EvaluatedIndividual:
    genes: Individual
    metrics: FitnessResult | None
    failure: str | None = None

    @property
    def score_key(self) -> tuple[float, float, float]:
        if self.metrics is None:
            return (-math.inf, -math.inf, -math.inf)
        return (
            self.metrics.fitness,
            self.metrics.f1_mean,
            self.metrics.precision_mean,
        )


@dataclass(frozen=True)
class GAResult:
    config: GAConfig
    best: EvaluatedIndividual
    history: list[dict[str, Any]]
    final_population: list[Individual]
    evaluation_failures: list[dict[str, Any]]


FitnessFunction = Callable[..., FitnessResult]


class GeneticAlgorithm:
    """GA com torneio, crossover uniforme, mutacao por gene e elitismo."""

    def __init__(
        self,
        config: GAConfig,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        fitness_function: FitnessFunction = evaluate_individual,
    ) -> None:
        config.validate()
        self.config = config
        self.X_train = X_train
        self.y_train = y_train
        self.fitness_function = fitness_function
        self.rng = random.Random(config.seed)
        self._cache: dict[str, EvaluatedIndividual] = {}
        self._failures: list[dict[str, Any]] = []

    def initial_population(self) -> list[Individual]:
        return [
            sample_individual(self.config.model, self.rng)
            for _ in range(self.config.population_size)
        ]

    def crossover(
        self, parent_a: Individual, parent_b: Individual
    ) -> tuple[Individual, Individual]:
        validate_individual(self.config.model, parent_a)
        validate_individual(self.config.model, parent_b)
        if self.rng.random() >= self.config.crossover_rate:
            return copy.deepcopy(parent_a), copy.deepcopy(parent_b)
        child_a: Individual = {}
        child_b: Individual = {}
        for gene in parent_a:
            if self.rng.random() < 0.5:
                child_a[gene], child_b[gene] = parent_a[gene], parent_b[gene]
            else:
                child_a[gene], child_b[gene] = parent_b[gene], parent_a[gene]
        return (
            repair_individual(self.config.model, child_a, self.rng),
            repair_individual(self.config.model, child_b, self.rng),
        )

    def mutate(self, individual: Individual) -> Individual:
        return mutate_individual(
            self.config.model, individual, self.config.mutation_rate, self.rng
        )

    @staticmethod
    def _cache_key(individual: Individual) -> str:
        return json.dumps(individual, sort_keys=True, separators=(",", ":"))

    def _evaluate(self, individual: Individual, generation: int) -> EvaluatedIndividual:
        key = self._cache_key(individual)
        if key in self._cache:
            cached = self._cache[key]
            return EvaluatedIndividual(copy.deepcopy(individual), cached.metrics, cached.failure)
        try:
            metrics = self.fitness_function(
                self.config.model,
                copy.deepcopy(individual),
                self.X_train,
                self.y_train,
                folds=self.config.folds,
                seed=self.config.seed,
            )
            evaluated = EvaluatedIndividual(copy.deepcopy(individual), metrics)
        except (CandidateEvaluationError, ValueError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            evaluated = EvaluatedIndividual(copy.deepcopy(individual), None, message)
            self._failures.append(
                {"generation": generation, "genes": copy.deepcopy(individual), "error": message}
            )
        self._cache[key] = evaluated
        return evaluated

    def _tournament(self, evaluated: list[EvaluatedIndividual]) -> Individual:
        contenders = self.rng.sample(evaluated, self.config.tournament_size)
        return copy.deepcopy(max(contenders, key=lambda item: item.score_key).genes)

    def _next_population(self, evaluated: list[EvaluatedIndividual]) -> list[Individual]:
        ranked = sorted(evaluated, key=lambda item: item.score_key, reverse=True)
        next_population = [copy.deepcopy(item.genes) for item in ranked[: self.config.elitism]]
        while len(next_population) < self.config.population_size:
            parent_a = self._tournament(evaluated)
            parent_b = self._tournament(evaluated)
            child_a, child_b = self.crossover(parent_a, parent_b)
            for child in (self.mutate(child_a), self.mutate(child_b)):
                if len(next_population) < self.config.population_size:
                    next_population.append(child)
        return next_population

    def run(self, initial_population: list[Individual] | None = None) -> GAResult:
        population = (
            copy.deepcopy(initial_population)
            if initial_population is not None
            else self.initial_population()
        )
        if len(population) != self.config.population_size:
            raise ValueError("A populacao inicial tem tamanho incorreto.")
        for individual in population:
            validate_individual(self.config.model, individual)

        history: list[dict[str, Any]] = []
        best_overall: EvaluatedIndividual | None = None
        for generation in range(self.config.generations):
            started = time.perf_counter()
            evaluated = [self._evaluate(item, generation) for item in population]
            successful = [item for item in evaluated if item.metrics is not None]
            if not successful:
                raise RuntimeError(f"Todos os individuos falharam na geracao {generation}.")
            generation_best = max(successful, key=lambda item: item.score_key)
            if best_overall is None or generation_best.score_key > best_overall.score_key:
                best_overall = EvaluatedIndividual(
                    copy.deepcopy(generation_best.genes), generation_best.metrics
                )
            fitness_values = [item.metrics.fitness for item in successful if item.metrics]
            metrics = generation_best.metrics
            assert metrics is not None
            history.append(
                {
                    "generation": generation,
                    "best_fitness_recall": metrics.fitness,
                    "mean_fitness_recall": statistics.fmean(fitness_values),
                    "std_fitness_recall": statistics.pstdev(fitness_values),
                    "best_precision": metrics.precision_mean,
                    "best_f1": metrics.f1_mean,
                    "best_recall_std": metrics.recall_std,
                    "best_false_negatives_total": metrics.false_negatives_total,
                    "best_genes": copy.deepcopy(generation_best.genes),
                    "elapsed_seconds": time.perf_counter() - started,
                    "evaluation_failures": len(evaluated) - len(successful),
                }
            )
            if generation + 1 < self.config.generations:
                population = self._next_population(evaluated)
                if len(population) != self.config.population_size:
                    raise AssertionError("Tamanho da populacao mudou entre geracoes.")

        assert best_overall is not None
        return GAResult(
            config=self.config,
            best=best_overall,
            history=history,
            final_population=copy.deepcopy(population),
            evaluation_failures=copy.deepcopy(self._failures),
        )


def config_as_dict(config: GAConfig) -> dict[str, Any]:
    return asdict(config)

