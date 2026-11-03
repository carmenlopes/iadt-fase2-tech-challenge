from __future__ import annotations

import json
import logging
import threading
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from breast_cancer_ga.interpretation import LLMProviderError, validate_interpretation_input
from breast_cancer_ga.web import create_server


def test_web_page_and_interpretation_route() -> None:
    def fake_interpreter(payload: object) -> str:
        validate_interpretation_input(payload)
        return "Explicação de teste."

    server = create_server(port=0, interpreter=fake_interpreter)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    payload = {
        "diagnosis": "benigno",
        "probabilities": {"benigno": 0.9, "maligno": 0.1},
        "metrics": {"recall": 0.92},
    }
    try:
        with urlopen(base_url) as response:
            page = response.read().decode()
        assert "Interpretação de resultado do modelo" in page
        assert "textContent" in page
        assert "Métricas agregadas (informe pelo menos uma; escala de 0 a 1)" in page

        request = Request(
            f"{base_url}/interpret",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            result = json.loads(response.read())
        assert result == {"interpretation": "Explicação de teste."}

        invalid = Request(
            f"{base_url}/interpret",
            data=b'{"diagnosis":"benigno"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(invalid)
        except HTTPError as exc:
            assert exc.code == 400
            assert "probabilities" in json.loads(exc.read())["error"]
        else:
            raise AssertionError("Payload inválido deveria retornar HTTP 400.")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_metrics_health_and_private_logs(caplog) -> None:
    server = create_server(port=0, interpreter=lambda payload: "ok")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with caplog.at_level(logging.INFO, logger="breast_cancer_ga.web"):
            req = Request(base + "/interpret", data=b'{"private":"secret"}',
                          headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req) as response:
                assert response.status == 200
            with urlopen(base + "/healthz") as response:
                assert json.loads(response.read()) == {"status": "ok"}
            with urlopen(base + "/metrics") as response:
                metrics = response.read().decode()
        assert 'interpreter_requests_total{route="interpret",status="200"} 1' in metrics
        assert 'interpreter_request_duration_seconds_count{route="interpret",status="200"} 1' in metrics
        assert "secret" not in metrics and "secret" not in caplog.text
        assert "route=interpret status=200" in caplog.text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_metrics_track_provider_errors_and_invalid_inputs(caplog) -> None:
    def failing_interpreter(payload: object) -> str:
        validate_interpretation_input(payload)
        raise LLMProviderError("falha simulada")

    server = create_server(port=0, interpreter=failing_interpreter)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    valid = {"diagnosis": "benigno", "probabilities": {"benigno": 0.9, "maligno": 0.1},
             "metrics": {"recall": 0.92}}
    try:
        with caplog.at_level(logging.INFO, logger="breast_cancer_ga.web"):
            for payload, expected in [({}, 400), (valid, 502)]:
                request = Request(base + "/interpret", data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
                try:
                    urlopen(request)
                except HTTPError as exc:
                    assert exc.code == expected
                else:
                    raise AssertionError("A solicitação deveria falhar.")
            with urlopen(base + "/metrics") as response:
                metrics = response.read().decode()
        for status in (400, 502):
            assert f'interpreter_requests_total{{route="interpret",status="{status}"}} 1' in metrics
            assert f'route=interpret status={status}' in caplog.text
        assert 'interpreter_active_requests 0' in metrics
        assert 'falha simulada' not in caplog.text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls"])
def test_web_rejects_incomplete_provider_responses(finish_reason, monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_URL", "https://provider.example.test/chat/completions")
    monkeypatch.setenv("LLM_API_KEY", "not-a-real-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    provider_body = {"choices": [{"message": {"content": "Texto parcial do provedor."},
                                  "finish_reason": finish_reason}]}
    server = create_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    payload = {"diagnosis": "benigno", "probabilities": {"benigno": 0.9, "maligno": 0.1},
               "metrics": {"recall": 0.92}}
    request = Request(f"http://127.0.0.1:{server.server_port}/interpret",
                      data=json.dumps(payload).encode(), method="POST",
                      headers={"Content-Type": "application/json"})
    try:
        with patch("breast_cancer_ga.interpretation.urlopen",
                   return_value=BytesIO(json.dumps(provider_body).encode())) as provider:
            with pytest.raises(HTTPError) as error:
                urlopen(request)
        assert error.value.code == 502
        result = json.loads(error.value.read())
        assert "incompleta" in result["error"]
        assert "interpretation" not in result
        assert "Texto parcial" not in json.dumps(result)
        assert provider.call_count == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
