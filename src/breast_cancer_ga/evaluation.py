"""Small, auditable LLM study with a persistent budget and human review."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Mapping
from urllib.parse import urlsplit

from . import interpretation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "interpretation_cases.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "llm"
CRITERIA = ("fidelity", "statistics", "uncertainty", "utility", "clarity")
CRITICAL = ("invented_results", "statistical_inversion", "confirmed_diagnosis", "patient_advice")
REVIEW_FIELDS = (
    "run_id", "case_id", "attempt_id", "response_sha256", "scenario", "input_json",
    "response", "expected", "forbidden", "reviewer_id", "reviewer_profile",
    *CRITERIA, *CRITICAL, "justification", "error_excerpt",
)
PROTOCOL = {
    "version": 1, "dev_cases": 2, "final_cases": 8, "generations_per_case": 1,
    "criteria": list(CRITERIA), "score_values": [0, 1, 2],
    "critical_flags": list(CRITICAL), "response_threshold": 8,
    "study_threshold": 0.9, "human_review_required": True,
}
LIMITATIONS = (
    "Amostra pequena de oito cenários finais, incluindo dados sintéticos.",
    "Uma geração por caso; o estudo não mede variabilidade entre gerações da mesma entrada.",
    "Revisão técnica e textual pela equipe; não houve revisão clínica especializada.",
    "As notas não constituem validação clínica nem demonstram generalização.",
)


class EvaluationError(ValueError):
    """Invalid study configuration or evidence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    """Atomic replacement: an interruption cannot partially reset the ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def study_lock(root: Path):
    """OS lock is released even on process death; two collectors cannot overspend."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".evaluation.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise EvaluationError("Outra operação de avaliação está em andamento.") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _run_path(root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id) or run_id == "budget":
        raise EvaluationError("run_id deve conter somente letras, números, _ e -.")
    return root / run_id


def _metrics(matrix: list) -> dict[str, float]:
    if (not isinstance(matrix, list) or len(matrix) != 2
            or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
            or any(type(x) is not int or x < 0 for row in matrix for x in row)):
        raise EvaluationError("Matriz deve conter quatro contagens inteiras não negativas.")
    (tn, fp), (fn, tp) = matrix
    if not tn + fp + fn + tp or not tp + fn or not tp + fp:
        raise EvaluationError("Matriz deve permitir definir accuracy, precision e recall.")
    return {"accuracy": (tn + tp) / (tn + fp + fn + tp), "precision": tp / (tp + fp),
            "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn)}


def load_cases(path: Path = DEFAULT_CASES, *, verify_sources: bool = True) -> dict:
    suite = read_json(path)
    if not isinstance(suite, dict) or suite.get("version") != 1 or not isinstance(suite.get("cases"), list):
        raise EvaluationError("Conjunto de casos inválido.")
    counts = {"dev": 0, "final": 0}
    ids, payloads = set(), set()
    for case in suite["cases"]:
        if not isinstance(case, dict) or not {"payload", "source"}.issubset(case):
            raise EvaluationError("Cada caso deve declarar payload e source.")
        if (not isinstance(case.get("id"), str) or not case["id"]
                or case["id"] in ids or case.get("split") not in counts):
            raise EvaluationError("Caso duplicado ou split inválido.")
        ids.add(case["id"])
        counts[case["split"]] += 1
        validated = interpretation.validate_interpretation_input(case["payload"]).to_payload()
        payload_hash = digest(validated)
        if payload_hash in payloads:
            raise EvaluationError("Payload duplicado entre casos de desenvolvimento/finais.")
        payloads.add(payload_hash)
        if (case.get("probabilities_source") != "synthetic" or not case.get("scenario")
                or any(not isinstance(case.get(key), list) or not case[key]
                       or any(not isinstance(x, str) or not x.strip() for x in case[key])
                       for key in ("expected", "forbidden"))):
            raise EvaluationError("Caso deve declarar origem, expectativas e erros a evitar.")
        source = case["source"]
        computed = _metrics(source["confusion_matrix"])
        for name, value in validated["metrics"].items():
            if not math.isclose(value, computed[name], abs_tol=1e-10):
                raise EvaluationError(f"Métrica inconsistente: {case['id']}/{name}.")
        probabilities = validated["probabilities"]
        if probabilities[validated["diagnosis"]] < max(probabilities.values()):
            raise EvaluationError("Classe prevista contradiz as probabilidades.")
        if source["kind"] not in ("synthetic", "project"):
            raise EvaluationError("Origem de métricas desconhecida.")
        if source["kind"] == "project" and verify_sources:
            source_path = (PROJECT_ROOT / source["path"]).resolve()
            if not source_path.is_relative_to(PROJECT_ROOT):
                raise EvaluationError("Fonte deve estar dentro do projeto.")
            comparison = read_json(source_path)["comparison"]
            original = next(x for x in comparison if x["model"] == source["model"])[source["variant"]]
            mapping = {"accuracy": "accuracy", "precision": "malignant_precision",
                       "recall": "malignant_recall", "f1": "malignant_f1"}
            if original["confusion_matrix"] != source["confusion_matrix"] or any(
                not math.isclose(value, original[mapping[name]], abs_tol=1e-10)
                for name, value in validated["metrics"].items()
            ):
                raise EvaluationError("Métricas do caso divergem do artefato original.")
    if counts != {"dev": 2, "final": 8}:
        raise EvaluationError("São necessários exatamente dois casos dev e oito finais.")
    return suite


def _ledger(root: Path) -> dict:
    path = root / "budget.json"
    if not path.exists():
        # Never silently reset a missing budget when evidence already exists.
        if any(root.glob("*/manifest.json")):
            raise EvaluationError("budget.json ausente; restaure o histórico antes de continuar.")
        return {"version": 1, "limit": 10, "attempts": [], "budget_changes": []}
    ledger = read_json(path)
    if (ledger.get("version") != 1 or type(ledger.get("limit")) is not int
            or ledger["limit"] < 10 or not isinstance(ledger.get("attempts"), list)
            or len({a["id"] for a in ledger["attempts"]}) != len(ledger["attempts"])):
        raise EvaluationError("Histórico de orçamento inválido.")
    changed = False
    for attempt in ledger["attempts"]:
        if attempt["status"] == "started":
            attempt.update(status="interrupted", finished_at=utc_now(),
                           error="Execução interrompida; tentativa permanece contabilizada.")
            changed = True
    if changed:
        write_json(path, ledger)
    return ledger


def _prepare(root: Path, run_id: str, suite: dict, ledger: dict) -> dict:
    run = _run_path(root, run_id)
    path = run / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        snapshot = read_json(run / "cases.json")
        if manifest.get("protocol") != PROTOCOL or manifest.get("protocol_sha256") != digest(PROTOCOL):
            raise EvaluationError("Protocolo de avaliação diverge da versão registrada.")
        if digest(snapshot) != manifest["cases_sha256"] or digest(suite) != manifest["cases_sha256"]:
            raise EvaluationError("Casos alterados: use nova execução; orçamento continua compartilhado.")
        if not path.parent.name == manifest["run_id"]:
            raise EvaluationError("Identificação da execução inválida.")
        return manifest
    manifest = {"version": 1, "run_id": run_id, "created_at": utc_now(),
                "cases_sha256": digest(suite), "cases_version": suite["version"],
                "protocol": PROTOCOL, "protocol_sha256": digest(PROTOCOL),
                "final_config": None, "attempts": [], "limitations": list(LIMITATIONS)}
    write_json(root / "budget.json", ledger)
    write_json(run / "cases.json", suite)
    write_json(path, manifest)
    return manifest


def prepare(root: Path, run_id: str, cases_path: Path = DEFAULT_CASES) -> dict:
    suite = load_cases(cases_path)
    with study_lock(root):
        ledger = _ledger(root)
        manifest = _prepare(root, run_id, suite, ledger)
        _save_run(root, manifest, ledger)
        return manifest


def _save_run(root: Path, manifest: dict, ledger: dict) -> None:
    manifest["attempts"] = [a for a in ledger["attempts"] if a["run_id"] == manifest["run_id"]]
    manifest["global_attempts_used"] = len(ledger["attempts"])
    manifest["global_attempt_limit"] = ledger["limit"]
    manifest["updated_at"] = utc_now()
    write_json(_run_path(root, manifest["run_id"]) / "manifest.json", manifest)


def increase_budget(root: Path, limit: int, reason: str) -> None:
    if not reason.strip():
        raise EvaluationError("Informe a autorização/justificativa para ampliar o orçamento.")
    with study_lock(root):
        ledger = _ledger(root)
        if type(limit) is not int or limit <= ledger["limit"]:
            raise EvaluationError("Novo teto deve ser maior que o teto atual.")
        ledger["budget_changes"].append({"at": utc_now(), "old": ledger["limit"],
                                         "new": limit, "reason": reason})
        ledger["limit"] = limit
        write_json(root / "budget.json", ledger)


def collect(root: Path, run_id: str, split: str, *, cases_path: Path = DEFAULT_CASES,
            environ: Mapping[str, str] | None = None, retry_failed: bool = False) -> dict:
    if split not in ("dev", "final"):
        raise EvaluationError("Split inválido.")
    suite = load_cases(cases_path)  # validate every case before spending anything
    environment = os.environ if environ is None else environ
    api_url, api_key, model = interpretation._provider_config(environment)
    # Persist only origin and an endpoint hash; never query strings or credentials.
    endpoint = urlsplit(api_url)
    config = {"requested_model": model, "provider_origin": f"{endpoint.scheme}://{endpoint.hostname}",
              "endpoint_sha256": digest(api_url), "temperature": interpretation.TEMPERATURE,
              "system_prompt": interpretation.SYSTEM_PROMPT,
              "prompt_sha256": digest(interpretation.SYSTEM_PROMPT)}
    with study_lock(root):
        ledger = _ledger(root)
        manifest = _prepare(root, run_id, suite, ledger)
        previous = [a for a in ledger["attempts"] if a["run_id"] == run_id]
        if split == "dev" and manifest["final_config"] is not None:
            raise EvaluationError("Coleta final iniciada; desenvolvimento está fechado nesta execução.")
        if split == "final":
            dev_ids = {c["id"] for c in suite["cases"] if c["split"] == "dev"}
            if not dev_ids.issubset({a["case_id"] for a in previous}):
                raise EvaluationError("Execute os dois casos de desenvolvimento antes da coleta final.")
            if manifest["final_config"] is None:
                manifest["final_config"] = {**config, "frozen_at": utc_now(),
                                             "cases_sha256": manifest["cases_sha256"]}
            elif any(manifest["final_config"][k] != v for k, v in config.items()):
                raise EvaluationError("Prompt, modelo ou endpoint mudou após congelar a coleta final.")
        _save_run(root, manifest, ledger)
        for case in suite["cases"]:
            if case["split"] != split:
                continue
            attempted = [a for a in previous if a["case_id"] == case["id"]]
            if any(a["status"] == "success" for a in attempted) or (attempted and not retry_failed):
                continue
            if len(ledger["attempts"]) >= ledger["limit"]:
                _save_run(root, manifest, ledger)
                raise EvaluationError(f"Teto de {ledger['limit']} tentativas atingido. Nenhuma chamada adicional.")
            attempt = {"id": uuid.uuid4().hex, "run_id": run_id, "case_id": case["id"],
                       "split": split, "started_at": utc_now(), "status": "started",
                       "case_sha256": digest(case), **config}
            ledger["attempts"].append(attempt)
            # Reserve durably BEFORE the network operation, including crashes/timeouts.
            write_json(root / "budget.json", ledger)
            _save_run(root, manifest, ledger)
            started = time.monotonic()
            try:
                result = interpretation.complete_interpretation(
                    case["payload"], environ=environment, system_prompt=config["system_prompt"])
                interpretation.validate_completion_result(result, api_key=api_key)
                attempt.update(status="success", response=result.text,
                               response_sha256=digest(result.text), returned_model=result.model,
                               response_id=result.response_id, finish_reason=result.finish_reason)
            except BaseException as exc:
                attempt["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "error"
                # Do not serialize exception text: provider/network errors may contain secrets.
                attempt["error"] = ("Execução interrompida." if attempt["status"] == "interrupted"
                                    else "Falha na chamada ao provedor; nenhuma resposta válida registrada.")
                status_code = getattr(exc.__cause__, "code", None)
                if type(status_code) is int and 100 <= status_code <= 599:
                    attempt["http_status"] = status_code
                attempt.update(finished_at=utc_now(), duration_seconds=time.monotonic() - started)
                write_json(root / "budget.json", ledger)
                _save_run(root, manifest, ledger)
                if attempt["status"] == "interrupted":
                    raise
                raise EvaluationError("Falha registrada e contabilizada. Coleta interrompida para preservar créditos.") from None
            attempt.update(finished_at=utc_now(), duration_seconds=time.monotonic() - started)
            write_json(root / "budget.json", ledger)
            _save_run(root, manifest, ledger)
            previous.append(attempt)
        return manifest


def _evidence(root: Path, run_id: str) -> tuple[dict, dict, list[dict]]:
    ledger = _ledger(root)
    run = _run_path(root, run_id)
    manifest = read_json(run / "manifest.json")
    if manifest.get("protocol") != PROTOCOL or manifest.get("protocol_sha256") != digest(PROTOCOL):
        raise EvaluationError("Protocolo de avaliação diverge da versão registrada.")
    suite = load_cases(run / "cases.json", verify_sources=False)
    if digest(suite) != manifest["cases_sha256"]:
        raise EvaluationError("Snapshot dos casos foi alterado.")
    cases = {c["id"]: c for c in suite["cases"]}
    attempts = [a for a in ledger["attempts"] if a["run_id"] == run_id]
    for attempt in attempts:
        if attempt["case_id"] not in cases or digest(cases[attempt["case_id"]]) != attempt["case_sha256"]:
            raise EvaluationError("Tentativa não corresponde ao caso congelado.")
        if attempt["status"] == "success" and digest(attempt["response"]) != attempt["response_sha256"]:
            raise EvaluationError("Resposta foi alterada após a coleta.")
        if attempt["split"] == "final":
            frozen = manifest["final_config"]
            if frozen is None or any(attempt[k] != frozen[k] for k in (
                "requested_model", "endpoint_sha256", "temperature", "prompt_sha256", "system_prompt"
            )):
                raise EvaluationError("Tentativa final diverge da configuração congelada.")
    _save_run(root, manifest, ledger)
    successes = [a for a in attempts if a["split"] == "final" and a["status"] == "success"]
    if len({a["case_id"] for a in successes}) != len(successes):
        raise EvaluationError("Mais de uma resposta final bem-sucedida para o mesmo caso.")
    return manifest, cases, successes


def _csv_safe(value: str) -> str:
    """Prevent spreadsheet execution while preserving quoted multiline text."""
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def _review_row(run_id: str, case: dict, attempt: dict) -> dict:
    row = {field: "" for field in REVIEW_FIELDS}
    row.update(run_id=run_id, case_id=case["id"], attempt_id=attempt["id"],
               response_sha256=attempt["response_sha256"], scenario=case["scenario"],
               input_json=json.dumps(case["payload"], ensure_ascii=False, sort_keys=True),
               response=attempt["response"], expected="; ".join(case["expected"]),
               forbidden="; ".join(case["forbidden"]))
    return {k: _csv_safe(v) for k, v in row.items()}


def export_review(root: Path, run_id: str, output: Path | None = None) -> Path:
    with study_lock(root):
        manifest, cases, successes = _evidence(root, run_id)
        output = output or _run_path(root, run_id) / "reviews.csv"
        with output.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for attempt in successes:
                writer.writerow(_review_row(run_id, cases[attempt["case_id"]], attempt))
        return output


def _read_reviews(path: Path | None, run_id: str, successes: list[dict], cases: dict) -> list[dict]:
    if path is None:
        return []
    by_id = {a["id"]: a for a in successes}
    rows, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(REVIEW_FIELDS):
            raise EvaluationError("Cabeçalhos da ficha foram alterados.")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise EvaluationError("Linha CSV incompleta ou com campos extras.")
            attempt = by_id.get(row["attempt_id"])
            if (attempt is None or row["attempt_id"] in seen or row["run_id"] != run_id
                    or row["case_id"] != attempt["case_id"]
                    or row["response_sha256"] != attempt["response_sha256"]
                    or row["response"] != _csv_safe(attempt["response"])):
                raise EvaluationError("Revisão duplicada, desconhecida ou ligada a outra resposta.")
            seen.add(row["attempt_id"])
            editable = ("reviewer_id", "reviewer_profile", *CRITERIA, *CRITICAL,
                        "justification", "error_excerpt")
            original = _review_row(run_id, cases[attempt["case_id"]], attempt)
            if any(row[k] != original[k] for k in REVIEW_FIELDS if k not in editable):
                raise EvaluationError("Entrada, resposta ou instruções da ficha foram alteradas.")
            if not any(row[k].strip() for k in editable):
                continue  # blank rows are pending, never zero scores
            if (any(row[k] not in ("0", "1", "2") for k in CRITERIA)
                    or any(row[k] not in ("0", "1") for k in CRITICAL)
                    or any(not row[k].strip() for k in ("reviewer_id", "reviewer_profile", "justification"))):
                raise EvaluationError("Preencha avaliador, perfil, notas 0–2, flags 0/1 e justificativa.")
            row["scores"] = {k: int(row[k]) for k in CRITERIA}
            row["critical_flags"] = {k: int(row[k]) for k in CRITICAL}
            has_error = any(v < 2 for v in row["scores"].values()) or any(row["critical_flags"].values())
            if has_error and (not row["error_excerpt"].strip()
                              or row["error_excerpt"] not in attempt["response"]):
                raise EvaluationError("Erros exigem um trecho literal da resposta como evidência.")
            row["total"] = sum(row["scores"].values())
            row["satisfactory"] = row["total"] >= PROTOCOL["response_threshold"] and not any(row["critical_flags"].values())
            rows.append(row)
    return rows


def report(root: Path, run_id: str, reviews_path: Path | None = None) -> dict:
    with study_lock(root):
        manifest, cases, successes = _evidence(root, run_id)
        reviews = _read_reviews(reviews_path, run_id, successes, cases)
        final_ids = {c["id"] for c in cases.values() if c["split"] == "final"}
        collected = {a["case_id"] for a in successes}
        reviewed = {r["case_id"] for r in reviews}
        complete = collected == final_ids and reviewed == final_ids
        satisfactory = sum(r["satisfactory"] for r in reviews)
        critical = sum(any(r["critical_flags"].values()) for r in reviews)
        ledger = _ledger(root)
        attempts = manifest["attempts"]
        summary = {
            "run_id": run_id, "generated_at": utc_now(), "evaluation_complete": complete,
            "quality_target_met": complete and satisfactory / len(final_ids) >= PROTOCOL["study_threshold"] and critical == 0,
            "expected_final": len(final_ids), "collected_final": len(successes),
            "reviewed_final": len(reviews), "pending_collection": sorted(final_ids - collected),
            "pending_review": sorted(collected - reviewed), "satisfactory": satisfactory,
            "satisfactory_fraction_expected": satisfactory / len(final_ids),
            "satisfactory_fraction_reviewed": satisfactory / len(reviews) if reviews else None,
            "critical_responses": critical,
            "mean_by_criterion": {k: mean(r["scores"][k] for r in reviews) if reviews else None for k in CRITERIA},
            "run_attempts": len(attempts), "global_attempts": len(ledger["attempts"]),
            "global_limit": ledger["limit"],
            "failures": [{"id": a["id"], "case_id": a["case_id"], "status": a["status"],
                          "error": a.get("error")} for a in attempts if a["status"] != "success"],
            "limitations": list(LIMITATIONS),
            "reviewers": sorted({(r["reviewer_id"], r["reviewer_profile"]) for r in reviews}),
            "reviews": reviews,
        }
        run = _run_path(root, run_id)
        write_json(run / "report.json", summary)
        write_json(run / "reviews.json", reviews)
        text = ["# Avaliação técnica e textual das interpretações", "",
                f"Execução: `{run_id}`. Gerado em {summary['generated_at']} (UTC).", "",
                f"Avaliação concluída: **{'sim' if complete else 'não'}**.",
                f"Meta de qualidade atingida: **{'sim' if summary['quality_target_met'] else 'não'}**.", "",
                f"Coleta final: {len(successes)}/8. Revisão: {len(reviews)}/8.",
                f"Respostas satisfatórias: {satisfactory}/8 previstas; {satisfactory}/{len(reviews)} revisadas.",
                f"Respostas com falha crítica: {critical}. Falhas de execução: {len(summary['failures'])}.",
                f"Tentativas globais: {len(ledger['attempts'])}/{ledger['limit']}; nesta execução: {len(attempts)}.", "",
                "## Protocolo", "",
                "Dois casos dev e oito finais distintos; uma geração por caso, temperatura 0.2.",
                "Cinco critérios 0–2. Resposta satisfatória: ≥8/10 e nenhuma falha crítica.",
                "Meta interna: ≥90% das oito respostas (8/8) e nenhuma falha crítica.",
                "A conclusão da avaliação independe de alcançar a meta. Notas ausentes são pendências.", "",
                "## Resultados por critério", "", "| Critério | Média (0–2) |", "|---|---|"]
        text.extend(f"| {k} | {value:.2f} |" if value is not None else f"| {k} | Pendente |"
                    for k, value in summary["mean_by_criterion"].items())
        text.extend(["", "## Casos e revisão", ""])
        review_map = {r["case_id"]: r for r in reviews}
        for case in cases.values():
            if case["split"] != "final":
                continue
            row = review_map.get(case["id"])
            text.extend([f"### {case['id']}: {case['scenario']}", "",
                         "Origem das métricas: " + case["source"]["kind"] + "; probabilidades sintéticas."])
            if row:
                text.extend([f"Nota: {row['total']}/10; satisfatória: {row['satisfactory']}.",
                             f"Avaliador: {row['reviewer_id']} ({row['reviewer_profile']}).",
                             "Justificativa: " + row["justification"],
                             "Trecho com erro: " + (row["error_excerpt"] or "Nenhum registrado.")])
            else:
                text.append("Pendente: " + ("revisão humana." if case["id"] in collected else "coleta real."))
            text.append("")
        text.extend(["## Falhas de execução", ""])
        text.extend(f"- {a['case_id']}: {a['status']} — {a['error']}" for a in summary["failures"])
        if not summary["failures"]:
            text.append("Nenhuma falha registrada; consulte a cobertura antes de inferir sucesso.")
        text.extend(["", "## Limitações", "", *("- " + x for x in LIMITATIONS), "",
                     "Evidências: manifest.json, cases.json, reviews.json e report.json nesta pasta.", ""])
        (run / "report.md").write_text("\n".join(text), encoding="utf-8")
        return summary
