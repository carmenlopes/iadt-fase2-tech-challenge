from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from breast_cancer_ga import evaluation as ev
from breast_cancer_ga import interpretation as llm

ENV = {"LLM_API_URL": "https://provider.example.test/v1/chat/completions?private=hidden",
       "LLM_API_KEY": "secret-never-persist", "LLM_MODEL": "requested-test-model"}
TEXT = 'Resumo do resultado\nExplicação, com acentuação e "aspas".\nLimitações\nSaída de modelo.'


@pytest.fixture
def provider(monkeypatch):
    calls = []

    def complete(payload, *, environ, system_prompt):
        llm.validate_interpretation_input(payload)
        assert set(payload) == {"diagnosis", "probabilities", "metrics"}
        calls.append(payload)
        return llm.CompletionResult(TEXT, "returned-test-model", "response-test-id", "stop")

    monkeypatch.setattr(llm, "complete_interpretation", complete)
    # Fail loudly if any evaluation test accidentally reaches a real endpoint.
    monkeypatch.setattr(llm, "urlopen", lambda *a, **k: pytest.fail("Unexpected network call"))
    return calls


def collect_all(root):
    ev.collect(root, "study", "dev", environ=ENV)
    ev.collect(root, "study", "final", environ=ENV)


def review_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_reviews(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ev.REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def scored(rows):
    for row in rows:
        row.update(reviewer_id="team-01", reviewer_profile="Engenharia e métricas de ML",
                   justification="Dados preservados e interpretação adequada neste exemplo de teste.")
        row.update({k: "2" for k in ev.CRITERIA})
        row.update({k: "0" for k in ev.CRITICAL})
    return rows


def test_cases_match_math_and_original_artifacts():
    suite = ev.load_cases()
    assert len(suite["cases"]) == 10
    assert sum(c["source"]["kind"] == "project" for c in suite["cases"]) == 4
    assert {c["payload"]["diagnosis"] for c in suite["cases"] if c["split"] == "final"} == {"benigno", "maligno"}


@pytest.mark.parametrize("mutation", ["duplicate", "split", "math", "extra", "source"])
def test_rejects_invalid_corpus_before_any_call(tmp_path, provider, mutation):
    suite = ev.load_cases()
    if mutation == "duplicate":
        suite["cases"][2]["payload"] = suite["cases"][0]["payload"]
    elif mutation == "split":
        suite["cases"][0]["split"] = "final"
    elif mutation == "math":
        suite["cases"][2]["payload"]["metrics"]["f1"] = 0.5
    elif mutation == "source":
        suite["cases"][2]["source"]["variant"] = "optimized"
    else:
        suite["cases"][0]["payload"]["patient_name"] = "not-allowed"
    path = tmp_path / "invalid.json"
    ev.write_json(path, suite)
    with pytest.raises(ValueError):
        ev.collect(tmp_path / "output", "study", "dev", cases_path=path, environ=ENV)
    assert not provider


def test_preflight_missing_configuration_costs_nothing(tmp_path, provider):
    with pytest.raises(llm.LLMConfigurationError):
        ev.collect(tmp_path, "study", "dev", environ={})
    assert not provider
    assert not (tmp_path / "budget.json").exists()


def test_prepare_and_report_are_offline_and_pending(tmp_path, provider):
    ev.prepare(tmp_path, "study")
    result = ev.report(tmp_path, "study")
    assert not provider
    assert result["collected_final"] == result["reviewed_final"] == 0
    assert not result["evaluation_complete"] and not result["quality_target_met"]
    assert result["mean_by_criterion"]["clarity"] is None


def test_ten_attempts_shared_across_splits_resumes_and_run_ids(tmp_path, provider):
    collect_all(tmp_path)
    ev.collect(tmp_path, "study", "final", environ=ENV)
    assert len(provider) == 10
    with pytest.raises(ev.EvaluationError, match="Teto de 10"):
        ev.collect(tmp_path, "other-study", "dev", environ=ENV)
    assert len(provider) == 10
    ledger = ev.read_json(tmp_path / "budget.json")
    assert len(ledger["attempts"]) == 10
    manifest = ev.read_json(tmp_path / "study" / "manifest.json")
    assert manifest["final_config"]["system_prompt"] == llm.SYSTEM_PROMPT
    assert {a["returned_model"] for a in manifest["attempts"]} == {"returned-test-model"}
    evidence = (tmp_path / "budget.json").read_text(encoding="utf-8")
    assert ENV["LLM_API_KEY"] not in evidence
    assert "private=hidden" not in evidence


def test_failure_and_explicit_retry_consume_budget_without_automatic_retries(tmp_path, provider, monkeypatch):
    original = llm.complete_interpretation

    def failure(*args, **kwargs):
        raise llm.LLMProviderError("Never save " + ENV["LLM_API_KEY"])

    monkeypatch.setattr(llm, "complete_interpretation", failure)
    with pytest.raises(ev.EvaluationError, match="contabilizada"):
        ev.collect(tmp_path, "study", "dev", environ=ENV)
    ledger = ev.read_json(tmp_path / "budget.json")
    assert len(ledger["attempts"]) == 1 and ledger["attempts"][0]["status"] == "error"
    assert ENV["LLM_API_KEY"] not in json.dumps(ledger)
    monkeypatch.setattr(llm, "complete_interpretation", original)
    ev.collect(tmp_path, "study", "dev", environ=ENV)
    assert len(provider) == 1  # failed case skipped, next case runs once
    ev.collect(tmp_path, "study", "dev", environ=ENV, retry_failed=True)
    assert len(provider) == 2
    with pytest.raises(ev.EvaluationError, match="Teto"):
        ev.collect(tmp_path, "study", "final", environ=ENV)
    assert len(ev.read_json(tmp_path / "budget.json")["attempts"]) == 10
    assert ev.report(tmp_path, "study")["collected_final"] == 7


def test_interruption_is_durable_and_resume_does_not_reset(tmp_path, provider, monkeypatch):
    original = llm.complete_interpretation
    monkeypatch.setattr(llm, "complete_interpretation", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        ev.collect(tmp_path, "study", "dev", environ=ENV)
    ledger = ev.read_json(tmp_path / "budget.json")
    assert len(ledger["attempts"]) == 1 and ledger["attempts"][0]["status"] == "interrupted"
    # Simulate abrupt process death before the completion status could be written.
    ledger["attempts"][0]["status"] = "started"
    ev.write_json(tmp_path / "budget.json", ledger)
    monkeypatch.setattr(llm, "complete_interpretation", original)
    ev.collect(tmp_path, "study", "dev", environ=ENV)
    ledger = ev.read_json(tmp_path / "budget.json")
    assert len(ledger["attempts"]) == 2 and ledger["attempts"][0]["status"] == "interrupted"
    assert len(provider) == 1


def test_final_requires_development_and_freezes_configuration(tmp_path, provider, monkeypatch):
    with pytest.raises(ev.EvaluationError, match="desenvolvimento"):
        ev.collect(tmp_path, "study", "final", environ=ENV)
    assert not provider
    collect_all(tmp_path)
    with pytest.raises(ev.EvaluationError, match="desenvolvimento"):
        ev.collect(tmp_path, "study", "dev", environ=ENV)
    with pytest.raises(ev.EvaluationError, match="modelo"):
        ev.collect(tmp_path, "study", "final", environ={**ENV, "LLM_MODEL": "different"})
    monkeypatch.setattr(llm, "SYSTEM_PROMPT", "changed")
    with pytest.raises(ev.EvaluationError, match="Prompt"):
        ev.collect(tmp_path, "study", "final", environ=ENV)
    assert len(provider) == 10


def test_changed_cases_and_missing_budget_do_not_reset_spend(tmp_path, provider):
    ev.collect(tmp_path, "study", "dev", environ=ENV)
    suite = ev.load_cases()
    suite["cases"][0]["expected"].append("New expectation")
    path = tmp_path / "changed.json"
    ev.write_json(path, suite)
    with pytest.raises(ev.EvaluationError, match="Casos alterados"):
        ev.collect(tmp_path, "study", "dev", cases_path=path, environ=ENV)
    (tmp_path / "budget.json").unlink()
    with pytest.raises(ev.EvaluationError, match="histórico"):
        ev.collect(tmp_path, "other", "dev", environ=ENV)
    assert len(provider) == 2


def test_os_lock_prevents_concurrent_collectors(tmp_path, provider):
    with ev.study_lock(tmp_path):
        with pytest.raises(ev.EvaluationError, match="andamento"):
            ev.collect(tmp_path, "study", "dev", environ=ENV)
    assert not provider


@pytest.mark.parametrize("result", [
    llm.CompletionResult("", "model", "id", "stop"),
    llm.CompletionResult("partial", "model", "id", "length"),
    llm.CompletionResult(ENV["LLM_API_KEY"], "model", "id", "stop"),
])
def test_empty_truncated_or_secret_response_is_failure(tmp_path, provider, monkeypatch, result):
    monkeypatch.setattr(llm, "complete_interpretation", lambda *a, **k: result)
    with pytest.raises(ev.EvaluationError):
        ev.collect(tmp_path, "study", "dev", environ=ENV)
    ledger = ev.read_json(tmp_path / "budget.json")
    assert len(ledger["attempts"]) == 1
    assert ledger["attempts"][0]["status"] == "error"
    assert "response" not in ledger["attempts"][0]


def test_csv_roundtrip_pending_and_complete_report(tmp_path, provider):
    collect_all(tmp_path)
    path = ev.export_review(tmp_path, "study")
    rows = review_rows(path)
    assert len(rows) == 8 and rows[0]["response"] == TEXT
    assert ev.report(tmp_path, "study", path)["reviewed_final"] == 0
    with pytest.raises(FileExistsError):
        ev.export_review(tmp_path, "study")  # never overwrite human work
    write_reviews(path, scored(rows))
    result = ev.report(tmp_path, "study", path)
    assert result["evaluation_complete"] and result["quality_target_met"]
    assert result["satisfactory"] == 8 and result["mean_by_criterion"]["clarity"] == 2
    assert (tmp_path / "study" / "reviews.json").exists()
    assert "8/8" in (tmp_path / "study" / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "invalid_score", "invalid_flag", "profile", "wrong_hash", "changed_input", "excerpt"])
def test_incomplete_or_invalid_reviews_never_pass(tmp_path, provider, mutation):
    collect_all(tmp_path)
    path = ev.export_review(tmp_path, "study")
    rows = scored(review_rows(path))
    if mutation == "missing":
        write_reviews(path, rows[:-1])
        result = ev.report(tmp_path, "study", path)
        assert not result["evaluation_complete"] and not result["quality_target_met"]
        assert result["satisfactory_fraction_expected"] == 7 / 8
        return
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "invalid_score":
        rows[0]["fidelity"] = "3"
    elif mutation == "invalid_flag":
        rows[0]["invented_results"] = "no"
    elif mutation == "profile":
        rows[0]["reviewer_profile"] = ""
    elif mutation == "wrong_hash":
        rows[0]["response_sha256"] = "wrong"
    elif mutation == "changed_input":
        rows[0]["input_json"] = "{}"
    else:
        rows[0]["clarity"] = "1"
        rows[0]["error_excerpt"] = "not in the actual response"
    write_reviews(path, rows)
    with pytest.raises(ev.EvaluationError):
        ev.report(tmp_path, "study", path)


def test_critical_failure_blocks_target_but_not_evaluation_completion(tmp_path, provider):
    collect_all(tmp_path)
    path = ev.export_review(tmp_path, "study")
    rows = scored(review_rows(path))
    rows[0].update(confirmed_diagnosis="1", error_excerpt="Saída de modelo.")
    write_reviews(path, rows)
    result = ev.report(tmp_path, "study", path)
    assert result["evaluation_complete"] and not result["quality_target_met"]
    assert result["critical_responses"] == 1 and result["satisfactory"] == 7


@pytest.mark.parametrize("notes,total,passes", [((2, 2, 2, 1, 1), 8, True), ((2, 2, 1, 1, 1), 7, False)])
def test_score_boundary_and_no_imputation(tmp_path, provider, notes, total, passes):
    collect_all(tmp_path)
    path = ev.export_review(tmp_path, "study")
    rows = scored(review_rows(path))
    rows[0].update({k: str(v) for k, v in zip(ev.CRITERIA, notes)})
    rows[0]["error_excerpt"] = "Saída de modelo."
    write_reviews(path, rows)
    result = ev.report(tmp_path, "study", path)
    assert result["reviews"][0]["total"] == total
    assert result["quality_target_met"] is passes


def test_explicit_budget_increase_preserves_history(tmp_path, provider):
    collect_all(tmp_path)
    with pytest.raises(ev.EvaluationError):
        ev.increase_budget(tmp_path, 11, "")
    ev.increase_budget(tmp_path, 12, "Explicit extra budget for test")
    ev.collect(tmp_path, "other", "dev", environ=ENV)
    ledger = ev.read_json(tmp_path / "budget.json")
    assert len(ledger["attempts"]) == 12 and len(ledger["budget_changes"]) == 1
    assert len(provider) == 12


def test_response_tampering_is_detected(tmp_path, provider):
    collect_all(tmp_path)
    ledger = ev.read_json(tmp_path / "budget.json")
    ledger["attempts"][-1]["response"] = "edited after generation"
    ev.write_json(tmp_path / "budget.json", ledger)
    with pytest.raises(ev.EvaluationError, match="alterada"):
        ev.report(tmp_path, "study")


def test_protocol_cannot_change_after_prepare(tmp_path, provider):
    ev.prepare(tmp_path, "study")
    path = tmp_path / "study" / "manifest.json"
    manifest = ev.read_json(path)
    manifest["protocol"]["response_threshold"] = 0
    ev.write_json(path, manifest)
    with pytest.raises(ev.EvaluationError, match="Protocolo"):
        ev.collect(tmp_path, "study", "dev", environ=ENV)
    with pytest.raises(ev.EvaluationError, match="Protocolo"):
        ev.report(tmp_path, "study")
    assert not provider


def test_provider_metadata_and_existing_text_contract():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"id": "generated-id", "model": "actual-model", "choices": [
                {"message": {"content": TEXT}, "finish_reason": "stop"}]}).encode()

    payload = ev.load_cases()["cases"][0]["payload"]
    with patch.object(llm, "urlopen", return_value=Response()) as opener:
        result = llm.complete_interpretation(payload, environ=ENV)
        assert result.model == "actual-model" and result.response_id == "generated-id"
        assert result.finish_reason == "stop"
        assert llm.interpret_diagnosis(payload, environ=ENV) == TEXT
        sent = json.loads(opener.call_args.args[0].data)
        assert sent["temperature"] == 0.2
        assert json.loads(sent["messages"][1]["content"]) == payload


def test_csv_formula_prefix_is_escaped(tmp_path, provider, monkeypatch):
    monkeypatch.setattr(llm, "complete_interpretation", lambda *a, **k: llm.CompletionResult(
        '=HYPERLINK("https://example.test")\nTexto', "model", "id", "stop"))
    collect_all(tmp_path)
    path = ev.export_review(tmp_path, "study")
    rows = scored(review_rows(path))
    assert rows[0]["response"].startswith("'=")
    write_reviews(path, rows)
    assert ev.report(tmp_path, "study", path)["evaluation_complete"]
