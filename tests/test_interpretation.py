from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import patch

import pytest

from breast_cancer_ga.interpretation import (
    LLMConfigurationError,
    LLMProviderError,
    SYSTEM_PROMPT,
    interpret_diagnosis,
    validate_interpretation_input,
)

VALID_PAYLOAD = {
    "diagnosis": "maligno",
    "probabilities": {"benigno": 0.12, "maligno": 0.88},
    "metrics": {"accuracy": 0.94, "recall": 0.91, "f1": 0.9},
}
ENVIRONMENT = {
    "LLM_API_URL": "https://llm.example.test/v1/chat/completions",
    "LLM_API_KEY": "test-key-not-real",
    "LLM_MODEL": "test-model",
}


def test_validates_only_aggregate_fields() -> None:
    result = validate_interpretation_input(VALID_PAYLOAD)
    assert result.to_payload() == VALID_PAYLOAD
    with pytest.raises(ValueError, match="outros campos"):
        validate_interpretation_input({**VALID_PAYLOAD, "features": {"radius": 17}})


@pytest.mark.parametrize(
    "payload",
    [
        {**VALID_PAYLOAD, "diagnosis": "cancer"},
        {**VALID_PAYLOAD, "probabilities": {"benigno": 0.4, "maligno": 0.4}},
        {**VALID_PAYLOAD, "metrics": {"recall": float("nan")}},
        {**VALID_PAYLOAD, "metrics": {"patient_id": "123"}},
    ],
)
def test_rejects_invalid_or_unapproved_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validate_interpretation_input(payload)


def test_requires_https_and_provider_credentials() -> None:
    with pytest.raises(LLMConfigurationError, match="LLM_API_URL"):
        interpret_diagnosis(VALID_PAYLOAD, environ={})
    with pytest.raises(LLMConfigurationError, match="HTTPS"):
        interpret_diagnosis(
            VALID_PAYLOAD,
            environ={**ENVIRONMENT, "LLM_API_URL": "http://llm.example.test/chat"},
        )


def test_sends_only_validated_output_and_uses_medical_safety_prompt() -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "Resumo do resultado\nTexto."}}]}
            ).encode()

    with patch("breast_cancer_ga.interpretation.urlopen", return_value=Response()) as opener:
        text = interpret_diagnosis(VALID_PAYLOAD, environ=ENVIRONMENT)

    request = opener.call_args.args[0]
    sent = json.loads(request.data)
    user_content = json.loads(sent["messages"][1]["content"])
    assert text == "Resumo do resultado\nTexto."
    assert user_content == VALID_PAYLOAD
    assert set(user_content) == {"diagnosis", "probabilities", "metrics"}
    assert SYSTEM_PROMPT.find("Não recomende tratamento") >= 0
    assert request.get_header("Authorization") == "Bearer test-key-not-real"


def test_provider_failure_has_explicit_message() -> None:
    from urllib.error import HTTPError

    failure = HTTPError(ENVIRONMENT["LLM_API_URL"], 401, "unauthorized", {}, BytesIO(b"secret"))
    with patch("breast_cancer_ga.interpretation.urlopen", side_effect=failure):
        with pytest.raises(RuntimeError, match="HTTP 401") as error:
            interpret_diagnosis(VALID_PAYLOAD, environ=ENVIRONMENT)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("finish_reason", ["stop", None])
def test_completed_responses_keep_existing_text_contract(finish_reason) -> None:
    body = {"choices": [{"message": {"content": " Texto completo. "},
                         "finish_reason": finish_reason}]}
    response = BytesIO(json.dumps(body).encode())
    with patch("breast_cancer_ga.interpretation.urlopen", return_value=response):
        assert interpret_diagnosis(VALID_PAYLOAD, environ=ENVIRONMENT) == "Texto completo."


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls"])
def test_incomplete_provider_responses_are_not_returned_as_explanations(finish_reason) -> None:
    body = {"choices": [{"message": {"content": "Texto parcial do provedor."},
                         "finish_reason": finish_reason}]}
    response = BytesIO(json.dumps(body).encode())
    with patch("breast_cancer_ga.interpretation.urlopen", return_value=response):
        with pytest.raises(LLMProviderError, match="incompleta") as error:
            interpret_diagnosis(VALID_PAYLOAD, environ=ENVIRONMENT)
    assert "Texto parcial" not in str(error.value)
