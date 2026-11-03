from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from breast_cancer_ga.fitness import FitnessResult
from breast_cancer_ga.ga import EvaluatedIndividual, GAResult

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("study_id", ["meu_estudo", "final_study_20261025"])
def test_study_id_matches_directory_comparison_manifest_and_run_configs(
    tmp_path, monkeypatch, study_id
):
    spec = importlib.util.spec_from_file_location(
        "run_final_study", ROOT / "scripts" / "run_final_study.py"
    )
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    for relative in (
        "dataset-breast-cancer/data.csv",
        "results/phase1_baseline/baseline_metrics.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)

    metrics = FitnessResult(0.9, 0.9, 0.0, 0.9, 0.0, 0.9, 0.0, 1.0, 5, [], [])

    def simulated_ga(config, *args):
        genes = (
            {"C": 1.0, "solver": "lbfgs", "penalty": "l2"}
            if config.model == "logistic_regression"
            else {"max_depth": 7, "min_samples_split": 2,
                  "min_samples_leaf": 1, "criterion": "gini"}
        )
        history = [
            {"generation": i, "best_fitness_recall": 0.9, "mean_fitness_recall": 0.9,
             "best_genes": genes}
            for i in range(config.generations)
        ]
        result = GAResult(config, EvaluatedIndividual(genes, metrics), history, [genes], [])
        return SimpleNamespace(run=lambda: result)

    # Exercise real artifact writing, without repeating expensive training/search.
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "GeneticAlgorithm", simulated_ga)
    monkeypatch.setattr(study, "score_on_test", lambda *args: {
        "accuracy": 104 / 114, "malignant_recall": 37 / 42, "malignant_precision": 37 / 42,
        "malignant_f1": 37 / 42, "false_negatives": 5,
        "confusion_matrix": [[67, 5], [5, 37]],
    })
    argv = ["run_final_study.py"]
    if study_id != study.STUDY_ID:
        argv += ["--study-id", study_id]
    monkeypatch.setattr(sys, "argv", argv)
    study.main()

    output = tmp_path / "results" / "ga" / study_id
    comparison = json.loads((output / "final_comparison.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "study_manifest.json").read_text(encoding="utf-8"))
    assert manifest["study_id"] == comparison["study_id"] == output.name == study_id
    assert manifest["runs_completed"] == 6
    assert manifest["selection_uses_test"] is False
    configs = list(output.glob("experiment_*/config.json"))
    assert len(configs) == 6
    assert all(json.loads(path.read_text(encoding="utf-8"))["study_id"] == study_id
               for path in configs)
