"""Run three GA configurations per model, select on training CV, then test once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from breast_cancer_ga.data import MALIGNANT_LABEL, make_original_split
from breast_cancer_ga.ga import GAConfig, GeneticAlgorithm
from breast_cancer_ga.models import build_pipeline
from breast_cancer_ga.reporting import save_ga_result, write_json


STUDY_ID = "final_study_20261025"
EXPERIMENTS = (
    {"name": "experiment_01", "population_size": 12, "generations": 8, "seed": 42,
     "crossover_rate": 0.8, "mutation_rate": 0.2, "elitism": 1, "tournament_size": 3},
    {"name": "experiment_02", "population_size": 20, "generations": 12, "seed": 43,
     "crossover_rate": 0.85, "mutation_rate": 0.25, "elitism": 2, "tournament_size": 3},
    {"name": "experiment_03", "population_size": 32, "generations": 16, "seed": 44,
     "crossover_rate": 0.9, "mutation_rate": 0.15, "elitism": 3, "tournament_size": 5},
)
MODELS = ("logistic_regression", "decision_tree")


def score_on_test(model: str, genes: dict[str, Any], split: Any) -> dict[str, Any]:
    pipeline = build_pipeline(model, genes, seed=42)
    pipeline.fit(split.X_train, split.y_train)
    prediction = pipeline.predict(split.X_test)
    tn, fp, fn, tp = confusion_matrix(
        split.y_test, prediction, labels=[0, MALIGNANT_LABEL]
    ).ravel()
    return {
        "accuracy": float(accuracy_score(split.y_test, prediction)),
        "malignant_recall": float(
            recall_score(split.y_test, prediction, pos_label=MALIGNANT_LABEL, zero_division=0)
        ),
        "malignant_precision": float(
            precision_score(split.y_test, prediction, pos_label=MALIGNANT_LABEL, zero_division=0)
        ),
        "malignant_f1": float(
            f1_score(split.y_test, prediction, pos_label=MALIGNANT_LABEL, zero_division=0)
        ),
        "false_negatives": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def make_analysis(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Análise dos resultados finais", "",
        "A escolha de hiperparâmetros foi feita exclusivamente pelo recall médio da classe maligna em validação cruzada no treino; F1 e precision foram usados como desempates. O conjunto de teste foi reservado até essa escolha e avaliado uma vez por modelo final.",
        "",
        "## Comparação no teste reservado", "",
        "| Modelo | Recall maligno (baseline → otimizado) | FN (baseline → otimizado) | Accuracy (baseline → otimizado) | Leitura |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        old, new = row["baseline"], row["optimized"]
        recall_delta = new["malignant_recall"] - old["malignant_recall"]
        fn_delta = new["false_negatives"] - old["false_negatives"]
        if recall_delta > 1e-12:
            recall_text = "Recall aumentou"
        elif recall_delta < -1e-12:
            recall_text = "Recall diminuiu"
        else:
            recall_text = "Recall permaneceu igual"
        if fn_delta < 0:
            fn_text = f"FN diminuíram em {-fn_delta}"
        elif fn_delta > 0:
            fn_text = f"FN aumentaram em {fn_delta}"
        else:
            fn_text = "FN permaneceram iguais"
        lines.append(
            f"| {row['model_label']} | {old['malignant_recall']:.4f} → {new['malignant_recall']:.4f} "
            f"({recall_delta:+.4f}) | {old['false_negatives']} → {new['false_negatives']} "
            f"({fn_delta:+d}) | {old['accuracy']:.4f} → {new['accuracy']:.4f} "
            f"({new['accuracy'] - old['accuracy']:+.4f}) | {recall_text}; {fn_text}. |"
        )
    lines.extend([
        "", "## Interpretação e limites", "",
        "Recall e falsos negativos devem ser lidos juntos: maior recall significa que uma fração maior dos casos malignos foi identificada, enquanto menos falsos negativos significa menos casos malignos classificados como benignos neste conjunto. Precision e accuracy mostram possíveis custos em falsos positivos e acerto geral. As diferenças observadas valem para este split fixo e não demonstram, isoladamente, desempenho clínico ou generalização para outras populações.",
        "",
        "Os resultados dos experimentos e as métricas por geração estão nas pastas de cada execução. A tabela compara o baseline reproduzido da Fase 1 com os candidatos escolhidos sem consultar o teste.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-id", default=STUDY_ID,
        help="Nome novo para o diretório em results/ga (o padrão é o estudo atual).",
    )
    study_id = parser.parse_args().study_id
    if not study_id or Path(study_id).name != study_id or study_id in {".", ".."}:
        raise ValueError("study-id deve ser um único nome de diretório.")
    study_dir = ROOT / "results" / "ga" / study_id
    if study_dir.exists() and any(study_dir.iterdir()):
        raise FileExistsError(f"Diretório do estudo já existe e não está vazio: {study_dir}")
    study_dir.mkdir(parents=True, exist_ok=True)
    split = make_original_split(ROOT / "dataset-breast-cancer" / "data.csv")
    all_runs: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}

    # All candidate search and selection below receive only X_train/y_train.
    for model in MODELS:
        for experiment in EXPERIMENTS:
            config_values = {key: value for key, value in experiment.items() if key != "name"}
            config = GAConfig(model=model, folds=5, **config_values)
            result = GeneticAlgorithm(config, split.X_train, split.y_train).run()
            run_name = f"{experiment['name']}_{model}"
            run_dir = study_dir / run_name
            save_ga_result(
                result,
                run_dir,
                run_metadata={
                    "study_id": study_id,
                    "experiment": experiment,
                    "dataset_sha256": split.dataset_sha256,
                    "train_rows": len(split.X_train),
                    "test_rows_reserved": len(split.X_test),
                    "test_used_by_ga": False,
                    "selection_metrics_source": "stratified_5_fold_cv_on_training_only",
                },
            )
            assert result.best.metrics is not None
            all_runs[model].append({
                "experiment": experiment,
                "run_dir": run_name,
                "genes": result.best.genes,
                "metrics": result.best.metrics.to_dict(),
                "score_key": list(result.best.score_key),
            })
            print(f"Concluído: {run_name}; melhor recall CV={result.best.metrics.recall_mean:.4f}")

    selected: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        # This ranking uses only cross-validation metrics from the training split.
        selected[model] = max(all_runs[model], key=lambda row: tuple(row["score_key"]))

    baseline_path = ROOT / "results" / "phase1_baseline" / "baseline_metrics.json"
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_by_model = {row["model"]: row for row in baseline_payload["models"]}

    # Final held-out evaluation: do not use these outcomes to revise selection.
    comparison_rows: list[dict[str, Any]] = []
    for model in MODELS:
        candidate = selected[model]
        metrics = score_on_test(model, candidate["genes"], split)
        baseline = baseline_by_model[model]
        baseline_metrics = {
            "accuracy": baseline["accuracy"],
            "malignant_recall": baseline["malignant_recall"],
            "malignant_precision": baseline["malignant_precision"],
            "malignant_f1": baseline["malignant_f1"],
            "false_negatives": baseline["false_negatives_malignant_as_benign"],
            "confusion_matrix": baseline["confusion_matrix"],
        }
        comparison_rows.append({
            "model": model,
            "model_label": "Regressão Logística" if model == "logistic_regression" else "Árvore de Decisão",
            "selection": {
                "experiment": candidate["experiment"],
                "run_dir": candidate["run_dir"],
                "genes": candidate["genes"],
                "cv_metrics": candidate["metrics"],
                "criterion": "recall_mean, then f1_mean, then precision_mean",
                "test_used_for_selection": False,
            },
            "baseline": baseline_metrics,
            "optimized": metrics,
        })

    payload = {
        "study_id": study_id,
        "dataset_sha256": split.dataset_sha256,
        "split": {"test_size": split.test_size, "seed": split.split_seed,
                  "train_rows": len(split.X_train), "test_rows": len(split.X_test)},
        "selection_protocol": "Best CV candidate per model across three distinct GA configurations; recall, then F1, then precision; test outcomes excluded.",
        "test_evaluations": 2,
        "experiments": all_runs,
        "comparison": comparison_rows,
    }
    write_json(study_dir / "final_comparison.json", payload)
    table_lines = [
        "# Comparação final: baseline e modelo otimizado", "",
        "Os candidatos otimizados foram selecionados por validação cruzada somente no treino, entre três configurações de GA por modelo. O teste reservado foi usado depois da seleção para a avaliação final.", "",
        "| Modelo | Versão | Accuracy | Precision maligna | Recall maligno | F1 maligno | FN | Matriz [[TN, FP], [FN, TP]] |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison_rows:
        for version, key in (("Original", "baseline"), ("Otimizado", "optimized")):
            m = row[key]
            table_lines.append(
                f"| {row['model_label']} | {version} | {m['accuracy']:.4f} | "
                f"{m['malignant_precision']:.4f} | {m['malignant_recall']:.4f} | "
                f"{m['malignant_f1']:.4f} | {m['false_negatives']} | `{json.dumps(m['confusion_matrix'])}` |"
            )
    table_lines.extend(["", "## Hiperparâmetros escolhidos", ""])
    for row in comparison_rows:
        table_lines.append(f"- **{row['model_label']}** — experimento `{row['selection']['run_dir']}`; genes: `{json.dumps(row['selection']['genes'], ensure_ascii=False, sort_keys=True)}`; Recall CV={row['selection']['cv_metrics']['recall_mean']:.4f}, F1 CV={row['selection']['cv_metrics']['f1_mean']:.4f}, Precision CV={row['selection']['cv_metrics']['precision_mean']:.4f}.")
    table_lines.extend(["", "Métricas originais vêm da reprodução da Fase 1 em `results/phase1_baseline/baseline_metrics.json`.", ""])
    (study_dir / "final_comparison.md").write_text("\n".join(table_lines), encoding="utf-8")
    (study_dir / "analysis.md").write_text(make_analysis(comparison_rows), encoding="utf-8")
    manifest = {
        "study_id": study_id,
        "distinct_ga_configurations": list(EXPERIMENTS),
        "models": list(MODELS),
        "runs_completed": 0,
        "status": "complete",
        "selection_uses_test": False,
        "final_test_evaluation_count": 2,
    }
    manifest["runs_completed"] = sum(len(runs) for runs in all_runs.values())
    write_json(study_dir / "study_manifest.json", manifest)
    print(f"Estudo concluído: {study_dir}")


if __name__ == "__main__":
    main()
