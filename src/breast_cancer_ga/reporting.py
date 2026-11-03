"""Persistencia dos artefatos de baseline e das buscas geneticas."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ga import GAResult, config_as_dict


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_evolution_plot(history: list[dict[str, Any]], path: str | Path) -> None:
    """Gera o grafico de evolucao a partir de um historico ja persistido."""
    generations = [row["generation"] for row in history]
    best = [row["best_fitness_recall"] for row in history]
    mean = [row["mean_fitness_recall"] for row in history]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(generations, best, marker="o", label="Melhor recall")
    axis.plot(generations, mean, marker="o", label="Recall medio")
    axis.set(xlabel="Geracao", ylabel="Recall maligno (CV)", title="Evolucao da busca")
    axis.set_xlim(min(generations) - 0.1, max(generations) + 0.1)
    axis.set_xticks(generations, labels=[str(value) for value in generations])
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_ga_result(
    result: GAResult,
    output_dir: str | Path,
    *,
    run_metadata: dict[str, Any] | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(
            f"Diretorio de resultado nao esta vazio: {output}. Use outro run-id."
        )
    write_json(
        output / "config.json",
        {"ga": config_as_dict(result.config), **(run_metadata or {})},
    )
    assert result.best.metrics is not None
    write_json(
        output / "best_individual.json",
        {"genes": result.best.genes, "metrics": result.best.metrics.to_dict()},
    )
    write_json(
        output / "summary.json",
        {
            "scope": "cross_validation_on_training_only",
            "objective": "maximize_mean_malignant_recall",
            "tie_break": ["mean_f1", "mean_precision"],
            "test_set_used": False,
            "best_genes": result.best.genes,
            "best_metrics": result.best.metrics.to_dict(),
            "evaluation_failures": result.evaluation_failures,
        },
    )
    history_rows = []
    for row in result.history:
        flat = dict(row)
        flat["best_genes"] = json.dumps(
            flat["best_genes"], ensure_ascii=False, sort_keys=True
        )
        history_rows.append(flat)
    with (output / "history.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)
    write_json(output / "history.json", result.history)

    save_evolution_plot(result.history, output / "evolution.png")

