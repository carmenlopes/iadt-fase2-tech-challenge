"""Collect and review LLM interpretations under a shared ten-attempt budget."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breast_cancer_ga.evaluation import (
    DEFAULT_CASES, DEFAULT_OUTPUT, EvaluationError, collect, export_review,
    increase_budget, prepare, report,
)
from breast_cancer_ga.interpretation import LLMConfigurationError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("prepare", "collect", "export-review", "report"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True, help="Identificador estável; reutilize na retomada")
        if name in ("prepare", "collect"):
            command.add_argument("--cases", type=Path, default=DEFAULT_CASES)
        if name == "collect":
            command.add_argument("--split", choices=("dev", "final"), required=True)
            command.add_argument("--retry-failed", action="store_true",
                                 help="Repetir falhas explicitamente; consome tentativas adicionais")
        if name == "export-review":
            command.add_argument("--output", type=Path)
        if name == "report":
            command.add_argument("--reviews", type=Path)
    budget = commands.add_parser("increase-budget", help="Somente após autorização de mais chamadas")
    budget.add_argument("--limit", type=int, required=True)
    budget.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "prepare":
            prepare(DEFAULT_OUTPUT, args.run, args.cases)
            print(f"Estudo {args.run} preparado sem chamadas à API.")
        elif args.operation == "collect":
            result = collect(DEFAULT_OUTPUT, args.run, args.split,
                             cases_path=args.cases, retry_failed=args.retry_failed)
            print(f"Coleta {args.split}: {result['global_attempts_used']}/{result['global_attempt_limit']} tentativas globais.")
        elif args.operation == "export-review":
            print(export_review(DEFAULT_OUTPUT, args.run, args.output))
        elif args.operation == "report":
            result = report(DEFAULT_OUTPUT, args.run, args.reviews)
            print(f"Coleta final: {result['collected_final']}/8; revisão: {result['reviewed_final']}/8.")
            print(f"Avaliação concluída: {result['evaluation_complete']}; meta atingida: {result['quality_target_met']}.")
        else:
            increase_budget(DEFAULT_OUTPUT, args.limit, args.reason)
            print("Ampliação explícita registrada no histórico do orçamento.")
    except (ValueError, LLMConfigurationError, OSError, KeyError, TypeError, StopIteration) as exc:
        # Never print arbitrary exception details from inputs or provider credentials.
        message = str(exc) if isinstance(exc, (EvaluationError, LLMConfigurationError)) else (
            "Não foi possível ler/gravar os artefatos; confira caminhos, formato e permissões.")
        print(message, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Execução interrompida; a tentativa permanece contabilizada.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
