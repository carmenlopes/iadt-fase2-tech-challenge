"""Reproduz e salva os dois baselines da Fase 1."""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from breast_cancer_ga.data import CLASS_NAMES, MALIGNANT_LABEL, make_original_split
from breast_cancer_ga.reporting import write_json


def metric_record(name: str, estimator, split, hyperparameters: dict) -> dict:
    estimator.fit(split.X_train, split.y_train)
    prediction = estimator.predict(split.X_test)
    tn, fp, fn, tp = confusion_matrix(split.y_test, prediction, labels=[0, 1]).ravel()
    return {
        "model": name,
        "procedure": "phase1_reproduced_test_baseline",
        "accuracy": float(accuracy_score(split.y_test, prediction)),
        "malignant_recall": float(recall_score(split.y_test, prediction, pos_label=MALIGNANT_LABEL)),
        "malignant_precision": float(precision_score(split.y_test, prediction, pos_label=MALIGNANT_LABEL)),
        "malignant_f1": float(f1_score(split.y_test, prediction, pos_label=MALIGNANT_LABEL)),
        "false_negatives_malignant_as_benign": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "confusion_matrix_axes": {
            "rows_true": [CLASS_NAMES[0], CLASS_NAMES[1]],
            "columns_predicted": [CLASS_NAMES[0], CLASS_NAMES[1]],
        },
        "hyperparameters": hyperparameters,
    }


def main() -> None:
    dataset = ROOT / "dataset-breast-cancer" / "data.csv"
    output = ROOT / "results" / "phase1_baseline"
    output.mkdir(parents=True, exist_ok=True)
    split = make_original_split(dataset)

    logistic_params = {
        "estimator": "LogisticRegression",
        "C": 1.0,
        "solver": "lbfgs",
        "penalty": "l2",
        "max_iter": 100,
        "random_state": None,
        "preprocessing": "StandardScaler fit on training data only",
    }
    logistic = Pipeline(
        [("preprocess", StandardScaler()), ("model", LogisticRegression())]
    )
    tree_params = {
        "estimator": "DecisionTreeClassifier",
        "max_depth": 7,
        "criterion": "gini",
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "selection": "5-fold StratifiedKFold on training data; recall; seed 42",
        "searched_max_depth": "1..8",
    }
    tree = DecisionTreeClassifier(max_depth=7, random_state=42)
    records = [
        metric_record("logistic_regression", logistic, split, logistic_params),
        metric_record("decision_tree", tree, split, tree_params),
    ]
    payload = {
        "scope": "held_out_test_baseline_only_not_ga_fitness",
        "dataset": {
            "path": "dataset-breast-cancer/data.csv",
            "sha256": split.dataset_sha256,
            "rows": len(split.X_train) + len(split.X_test),
            "features": split.X_train.shape[1],
            "malignant_source_value": "M",
            "malignant_encoded_value": MALIGNANT_LABEL,
            "benign_source_value": "B",
            "benign_encoded_value": 0,
        },
        "split": {
            "method": "train_test_split stratified by target",
            "test_size": split.test_size,
            "random_state": split.split_seed,
            "train_rows": len(split.X_train),
            "test_rows": len(split.X_test),
            "train_indices": split.train_indices,
            "test_indices": split.test_indices,
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "models": records,
    }
    write_json(output / "baseline_metrics.json", payload)

    with (output / "baseline_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = [
            "model", "accuracy", "malignant_recall", "malignant_precision",
            "malignant_f1", "false_negatives_malignant_as_benign", "tn", "fp", "fn", "tp",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            tn, fp = record["confusion_matrix"][0]
            fn, tp = record["confusion_matrix"][1]
            writer.writerow({
                **{key: record[key] for key in fields[:6]},
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            })

    table = [
        "# Referencia reproduzida da Fase 1",
        "",
        "Estas sao metricas no teste reservado e nao sao fitness da busca genetica.",
        "",
        "| Modelo | Accuracy | Recall maligno | Precision maligna | F1 maligno | FN | Matriz [[TN, FP], [FN, TP]] |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        table.append(
            f"| {record['model']} | {record['accuracy']:.4f} | "
            f"{record['malignant_recall']:.4f} | {record['malignant_precision']:.4f} | "
            f"{record['malignant_f1']:.4f} | {record['false_negatives_malignant_as_benign']} | "
            f"{json.dumps(record['confusion_matrix'])} |"
        )
    (output / "reference_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

