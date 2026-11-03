"""Validated, privacy-conscious LLM interpretation of model outputs."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

METRIC_NAMES = {"accuracy", "precision", "recall", "f1"}
DIAGNOSIS_NAMES = {"benigno", "maligno"}
PROBABILITY_NAMES = {"benigno", "maligno"}
MAX_PROBABILITY_SUM_ERROR = 0.02
TEMPERATURE = 0.2


class LLMProviderError(RuntimeError):
    """Error returned when provider configuration or completion fails."""


class LLMConfigurationError(LLMProviderError):
    """Missing or invalid provider configuration."""


@dataclass(frozen=True)
class InterpretationInput:
    diagnosis: str
    probabilities: dict[str, float]
    metrics: dict[str, float]

    def to_payload(self) -> dict[str, object]:
        return {
            "diagnosis": self.diagnosis,
            "probabilities": self.probabilities,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class CompletionResult:
    text: str
    model: str | None
    response_id: str | None
    finish_reason: str | None


def validate_completion_result(result: CompletionResult, *, api_key: str) -> None:
    """Apply the same response checks to the web interface and study collector."""
    if not isinstance(result.text, str) or not result.text.strip():
        raise LLMProviderError("O provedor LLM retornou uma interpretação vazia.")
    if result.finish_reason not in (None, "stop"):
        raise LLMProviderError("O provedor LLM retornou uma resposta interrompida ou incompleta.")
    if any(api_key in value for value in (result.text, result.model, result.response_id)
           if isinstance(value, str)):
        raise LLMProviderError("Resposta contém credencial; conteúdo descartado.")


SYSTEM_PROMPT = """\
Você explica resultados de modelos de classificação relacionados a câncer de mama \
em português claro para profissionais de saúde. Sua função é somente explicar os \
dados fornecidos, não diagnosticar uma pessoa nem orientar decisões de cuidado.

Regras obrigatórias:
- Trate a classificação e as probabilidades como saída de um modelo, não como \
diagnóstico clínico confirmado.
- Explique de forma concisa o que a classe e as métricas informadas permitem \
concluir; neste projeto, precision, recall e F1 referem-se à classe maligna, e \
accuracy é global. Não invente métricas, contexto, causas ou dados ausentes.
- Indique um ou dois pontos práticos para a equipe revisar sobre desempenho e \
validação do modelo, por exemplo o equilíbrio entre recall e precision. Não \
transforme esses pontos em recomendações para um paciente.
- Indique incerteza, limitações do modelo e que a interpretação deve ser \
considerada junto à avaliação profissional e ao contexto clínico.
- Não recomende tratamento, exames, procedimentos ou condutas para um paciente.
- Não revele nem infira dados pessoais. Se faltarem elementos, diga isso claramente.
- Organize a resposta nas seções "Resumo do resultado", "Leitura das métricas" e \
"Pontos de revisão do modelo" e "Limitações". Seja objetivo, cuidadoso e não use \
linguagem alarmista.
"""


def _unit_interval(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} deve ser um número entre 0 e 1.")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
        raise ValueError(f"{field} deve ser um número finito entre 0 e 1.")
    return numeric


def validate_interpretation_input(payload: object) -> InterpretationInput:
    """Accept only a classification, its class probabilities, and aggregate metrics."""
    if not isinstance(payload, dict):
        raise ValueError("O corpo da solicitação deve ser um objeto JSON.")
    if set(payload) != {"diagnosis", "probabilities", "metrics"}:
        raise ValueError(
            "Informe somente diagnosis, probabilities e metrics; "
            "outros campos não são aceitos."
        )

    diagnosis = payload["diagnosis"]
    if not isinstance(diagnosis, str) or diagnosis not in DIAGNOSIS_NAMES:
        raise ValueError("diagnosis deve ser 'benigno' ou 'maligno'.")

    probabilities = payload["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != PROBABILITY_NAMES:
        raise ValueError(
            "probabilities deve conter apenas benigno e maligno, ambos entre 0 e 1."
        )
    validated_probabilities = {
        name: _unit_interval(probabilities[name], f"probabilities.{name}")
        for name in sorted(PROBABILITY_NAMES)
    }
    if abs(sum(validated_probabilities.values()) - 1.0) > MAX_PROBABILITY_SUM_ERROR:
        raise ValueError("As probabilidades de benigno e maligno devem somar aproximadamente 1.")

    metrics = payload["metrics"]
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Informe ao menos uma métrica agregada.")
    if not set(metrics).issubset(METRIC_NAMES):
        raise ValueError("Métricas permitidas: accuracy, precision, recall e f1.")
    validated_metrics = {
        name: _unit_interval(value, f"metrics.{name}")
        for name, value in sorted(metrics.items())
    }
    return InterpretationInput(diagnosis, validated_probabilities, validated_metrics)


def _provider_config(environ: Mapping[str, str]) -> tuple[str, str, str]:
    api_url = environ.get("LLM_API_URL", "").strip()
    api_key = environ.get("LLM_API_KEY", "").strip()
    model = environ.get("LLM_MODEL", "").strip()
    if not api_url or not api_key or not model:
        raise LLMConfigurationError(
            "Configure LLM_API_URL, LLM_API_KEY e LLM_MODEL no ambiente."
        )
    try:
        parsed_url = urlsplit(api_url)
    except ValueError as exc:
        raise LLMConfigurationError("LLM_API_URL é inválida.") from exc
    if (
        parsed_url.scheme != "https"
        or not parsed_url.netloc
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.fragment
    ):
        raise LLMConfigurationError(
            "LLM_API_URL deve ser uma URL HTTPS sem credenciais ou fragmento."
        )
    return api_url, api_key, model


def complete_interpretation(
    payload: object,
    *,
    environ: Mapping[str, str] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> CompletionResult:
    """Send only validated aggregate output to an OpenAI-compatible HTTPS endpoint."""
    result = validate_interpretation_input(payload)
    api_url, api_key, model = _provider_config(os.environ if environ is None else environ)
    request_payload = {
        "model": model,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(result.to_payload(), ensure_ascii=False),
            },
        ],
    }
    request = Request(
        api_url,
        data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            response_body = response.read()
    except HTTPError as exc:
        raise LLMProviderError(
            f"O provedor LLM respondeu com HTTP {exc.code}."
        ) from exc
    except (TimeoutError, URLError) as exc:
        raise LLMProviderError(
            "Não foi possível conectar ao provedor LLM; verifique a rede e tente novamente."
        ) from exc

    try:
        decoded = json.loads(response_body)
        content = decoded["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError) as exc:
        raise LLMProviderError("O provedor LLM retornou uma resposta inválida.") from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMProviderError("O provedor LLM retornou uma interpretação vazia.")
    completion = CompletionResult(
        content.strip(),
        decoded.get("model") if isinstance(decoded.get("model"), str) else None,
        decoded.get("id") if isinstance(decoded.get("id"), str) else None,
        decoded["choices"][0].get("finish_reason"),
    )
    validate_completion_result(completion, api_key=api_key)
    return completion


def interpret_diagnosis(
    payload: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Keep the web interface's existing string-returning contract."""
    return complete_interpretation(payload, environ=environ).text
