"""Minimal local web interface for LLM-generated model explanations."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from .interpretation import (
    LLMConfigurationError,
    LLMProviderError,
    interpret_diagnosis,
)

MAX_REQUEST_BYTES = 16_384
LOGGER = logging.getLogger("breast_cancer_ga.web")


class ServiceMetrics:
    """Per-process counters; never store request bodies or client identifiers."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: dict[tuple[str, int], int] = defaultdict(int)
        self.duration: dict[tuple[str, int], float] = defaultdict(float)
        self.active = 0

    def begin(self) -> float:
        with self.lock:
            self.active += 1
        return time.monotonic()

    def end(self, route: str, status: int, start: float) -> None:
        elapsed = time.monotonic() - start
        with self.lock:
            self.active -= 1
            self.requests[route, status] += 1
            self.duration[route, status] += elapsed
        LOGGER.info('request route=%s status=%s duration_seconds=%.6f', route, status, elapsed)

    def render(self) -> bytes:
        with self.lock:
            lines = [
                '# HELP interpreter_active_requests Requests currently being handled.',
                '# TYPE interpreter_active_requests gauge',
                f'interpreter_active_requests {self.active}',
                '# HELP interpreter_requests_total Completed requests.',
                '# TYPE interpreter_requests_total counter',
            ]
            for (route, status), count in sorted(self.requests.items()):
                lines.append(f'interpreter_requests_total{{route="{route}",status="{status}"}} {count}')
            lines += ['# HELP interpreter_request_duration_seconds Total request duration in seconds.',
                      '# TYPE interpreter_request_duration_seconds summary']
            for (route, status), count in sorted(self.requests.items()):
                labels = f'route="{route}",status="{status}"'
                lines.append(f'interpreter_request_duration_seconds_sum{{{labels}}} {self.duration[route, status]:.6f}')
                lines.append(f'interpreter_request_duration_seconds_count{{{labels}}} {count}')
        return ('\n'.join(lines) + '\n').encode('utf-8')

PAGE = """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interpretação de resultado do modelo</title>
  <style>
    body { font: 16px/1.5 system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; color: #17212b; }
    fieldset { border: 1px solid #aab4bf; border-radius: .5rem; margin: 1rem 0; padding: 1rem; }
    label { display: block; margin: .65rem 0 .2rem; }
    input, select, button { font: inherit; padding: .45rem; }
    input, select { max-width: 18rem; width: 100%; }
    button { margin-top: 1rem; cursor: pointer; }
    #result { white-space: pre-wrap; border-left: 4px solid #3676a8; padding: .75rem 1rem; min-height: 2rem; }
    .notice { background: #f2f5f8; padding: .75rem; }
  </style>
</head>
<body>
  <h1>Interpretação de resultado do modelo</h1>
  <p class="notice">Demonstração educativa. A saída do modelo não é diagnóstico clínico.</p>
  <form id="interpretation-form">
    <fieldset>
      <legend>Classificação e probabilidades</legend>
      <label for="diagnosis">Classe prevista</label>
      <select id="diagnosis" name="diagnosis">
        <option value="benigno">Benigno</option>
        <option value="maligno">Maligno</option>
      </select>
      <label for="prob-benigno">Probabilidade de benigno</label>
      <input id="prob-benigno" name="prob-benigno" type="number" min="0" max="1" step="0.001" value="0.5" required>
      <label for="prob-maligno">Probabilidade de maligno</label>
      <input id="prob-maligno" name="prob-maligno" type="number" min="0" max="1" step="0.001" value="0.5" required>
    </fieldset>
    <fieldset>
      <legend>Métricas agregadas (informe pelo menos uma; escala de 0 a 1)</legend>
      <p>Preencha uma ou mais métricas. Os demais campos podem ficar em branco.</p>
      <label for="accuracy">Accuracy</label>
      <input id="accuracy" name="accuracy" type="number" min="0" max="1" step="0.001">
      <label for="precision">Precision da classe maligna</label>
      <input id="precision" name="precision" type="number" min="0" max="1" step="0.001">
      <label for="recall">Recall da classe maligna</label>
      <input id="recall" name="recall" type="number" min="0" max="1" step="0.001">
      <label for="f1">F1-score da classe maligna</label>
      <input id="f1" name="f1" type="number" min="0" max="1" step="0.001">
    </fieldset>
    <button id="submit" type="submit">Gerar explicação</button>
  </form>
  <h2>Interpretação</h2>
  <div id="result" role="status" aria-live="polite">A interpretação aparecerá aqui.</div>
  <script>
    const form = document.getElementById("interpretation-form");
    const result = document.getElementById("result");
    const button = document.getElementById("submit");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const values = new FormData(form);
      const metrics = {};
      for (const name of ["accuracy", "precision", "recall", "f1"]) {
        const value = values.get(name);
        if (value !== "") metrics[name] = Number(value);
      }
      const payload = {
        diagnosis: values.get("diagnosis"),
        probabilities: {
          benigno: Number(values.get("prob-benigno")),
          maligno: Number(values.get("prob-maligno"))
        },
        metrics
      };
      button.disabled = true;
      result.textContent = "Solicitando interpretação...";
      try {
        const response = await fetch("/interpret", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || "Falha ao gerar interpretação.");
        result.textContent = body.interpretation;
      } catch (error) {
        result.textContent = `Erro: ${error.message}`;
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    interpreter: Callable[[object], str] = interpret_diagnosis,
) -> ThreadingHTTPServer:
    metrics = ServiceMetrics()

    class Handler(BaseHTTPRequestHandler):
        def send_response(self, code: int, message: str | None = None) -> None:
            self._metric_status = code
            super().send_response(code, message)

        def _measured(self, route: str, action: Callable[[], None]) -> None:
            start = metrics.begin()
            self._metric_status = 500
            try:
                action()
            finally:
                metrics.end(route, self._metric_status, start)

        def _send_json(self, status: int, payload: dict[str, str]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send_json(200, {"status": "ok"})
                return
            if self.path == "/metrics":
                body = metrics.render()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._measured("page" if self.path == "/" else "other", self._get_page)

        def _get_page(self) -> None:
            if self.path != "/":
                self.send_error(404)
                return
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            self._measured("interpret" if self.path == "/interpret" else "other", self._post)

        def _post(self) -> None:
            if self.path != "/interpret":
                self._send_json(404, {"error": "Rota não encontrada."})
                return
            if self.headers.get_content_type() != "application/json":
                self._send_json(415, {"error": "Envie o conteúdo como application/json."})
                return
            try:
                content_length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send_json(400, {"error": "Content-Length inválido."})
                return
            if content_length < 0 or content_length > MAX_REQUEST_BYTES:
                self._send_json(413, {"error": "Solicitação muito grande."})
                return
            try:
                payload = json.loads(self.rfile.read(content_length))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send_json(400, {"error": "Corpo JSON inválido."})
                return
            try:
                explanation = interpreter(payload)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except LLMConfigurationError as exc:
                self._send_json(503, {"error": str(exc)})
                return
            except LLMProviderError as exc:
                self._send_json(502, {"error": str(exc)})
                return
            self._send_json(200, {"interpretation": explanation})

        def log_message(self, format: str, *args: object) -> None:
            # Keep request payloads and client details out of routine application logs.
            return

    return ThreadingHTTPServer((host, port), Handler)
