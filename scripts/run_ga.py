"""Executa uma busca genetica sem tocar no conjunto de teste."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breast_cancer_ga.data import make_original_split
from breast_cancer_ga.ga import GAConfig, GeneticAlgorithm
from breast_cancer_ga.reporting import save_ga_result
import pandas as pd
import sklearn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logistic_regression", "decision_tree"], required=True)
    parser.add_argument("--run-id", required=True, help="Subdiretorio novo em results/ga")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--elitism", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tournament-size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = make_original_split(ROOT / "dataset-breast-cancer" / "data.csv")
    config = GAConfig(
        model=args.model,
        seed=args.seed,
        population_size=args.population_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elitism=args.elitism,
        folds=args.folds,
        tournament_size=args.tournament_size,
    )
    # Somente X_train/y_train sao entregues ao GA: o teste fica fora da API de fitness.
    result = GeneticAlgorithm(config, split.X_train, split.y_train).run()
    output = ROOT / "results" / "ga" / args.run_id
    save_ga_result(
        result,
        output,
        run_metadata={
            "dataset": {
                "path": "dataset-breast-cancer/data.csv",
                "sha256": split.dataset_sha256,
                "train_rows": len(split.X_train),
                "test_rows_reserved": len(split.X_test),
                "features": split.X_train.shape[1],
                "malignant_source_value": "M",
                "malignant_encoded_value": 1,
                "benign_source_value": "B",
                "benign_encoded_value": 0,
            },
            "split": {
                "method": "train_test_split stratified by target",
                "test_size": split.test_size,
                "random_state": split.split_seed,
                "test_used_by_ga": False,
            },
            "environment": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
    )
    assert result.best.metrics is not None
    print(json.dumps({"output": str(output), "best_genes": result.best.genes,
                      "best_metrics": result.best.metrics.to_dict()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

