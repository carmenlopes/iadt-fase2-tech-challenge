# API local de interpretação de resultados

A API recebe uma saída de classificação já informada e pede a uma LLM que a explique. Ela não treina modelos nem executa predições a partir de atributos. O servidor usa `ThreadingHTTPServer`, da biblioteca padrão do Python, e escuta em `http://127.0.0.1:8000` por padrão.

## Iniciar o servidor

Na raiz do repositório, configure `LLM_API_URL`, `LLM_API_KEY` e `LLM_MODEL` no ambiente e execute:

```powershell
.\.venv\Scripts\python.exe scripts\serve_interpreter.py
```

As opções `--host` e `--port` alteram o endereço e a porta. A API local usa HTTP; a chamada do servidor ao provedor exige HTTPS. A chave fica no servidor. As [instruções de configuração](llm_interpretation.md) descrevem as variáveis.

## Rotas

| Método e caminho | Resposta de sucesso | Finalidade |
|---|---|---|
| `GET /` | 200, `text/html; charset=utf-8` | Exibe o formulário |
| `POST /interpret` | 200, `application/json; charset=utf-8` | Solicita uma explicação ao provedor |
| `GET /healthz` | 200, `application/json; charset=utf-8` | Retorna `{"status":"ok"}` se o processo responde |
| `GET /metrics` | 200, `text/plain; version=0.0.4; charset=utf-8` | Expõe contadores locais de requisições e duração |

Os caminhos são comparados literalmente. Acrescentar parâmetros de consulta ou uma barra final não identifica a mesma rota. `/healthz` não testa credenciais, modelo ou disponibilidade do provedor. `/healthz` e `/metrics` não entram na contagem e no logging rotineiro das requisições.

## Contrato de POST /interpret

Envie `Content-Type: application/json` e um corpo JSON com exatamente os três campos abaixo. O tamanho máximo é de **16.384 bytes**.

| Campo | Obrigatoriedade e valores |
|---|---|
| `diagnosis` | String obrigatória: `"benigno"` ou `"maligno"` |
| `probabilities` | Objeto obrigatório com exatamente `benigno` e `maligno` |
| `probabilities.benigno` e `.maligno` | Números finitos entre 0 e 1; a soma deve diferir de 1 em no máximo 0,02 |
| `metrics` | Objeto obrigatório e não vazio, com uma ou mais das chaves `accuracy`, `precision`, `recall`, `f1` |
| Valores de `metrics` | Números finitos entre 0 e 1; precision, recall e F1 se referem à classe maligna |

Booleanos, strings numéricas, valores nulos, números fora do intervalo e campos extras são rejeitados. O campo `diagnosis` é o nome técnico da classe informada; não representa confirmação clínica.

O formulário informa que pelo menos uma métrica é necessária; os campos restantes podem ficar em branco. A validação geral não verifica consistência entre classe e probabilidades nem recalcula métricas para comprovar sua origem. Essas verificações adicionais existem nos casos versionados do coletor de avaliação.

### Requisição completa

Este exemplo usa o caso **sintético** `dev_maligno`. Executar a requisição com credenciais válidas faz uma chamada real ao provedor. O formulário e esta rota não usam o teto de dez tentativas do coletor.

```http
POST /interpret HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json
Content-Length: 139

{"diagnosis":"maligno","probabilities":{"benigno":0.15,"maligno":0.85},"metrics":{"accuracy":0.8,"precision":0.75,"recall":0.75,"f1":0.75}}
```

No PowerShell, prefira deixar o cliente calcular `Content-Length`:

```powershell
$payload = @{
    diagnosis = "maligno"
    probabilities = @{ benigno = 0.15; maligno = 0.85 }
    metrics = @{ accuracy = 0.8; precision = 0.75; recall = 0.75; f1 = 0.75 }
} | ConvertTo-Json -Depth 3

Invoke-RestMethod -Uri "http://127.0.0.1:8000/interpret" `
    -Method Post -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($payload))
```

### Resposta de sucesso

O JSON possui uma chave, `interpretation`, cujo valor é o texto. Este exemplo é **ilustrativo**, não uma resposta coletada de uma LLM real:

```http
HTTP/1.0 200 OK
Content-Type: application/json; charset=utf-8
Cache-Control: no-store

{"interpretation":"Resumo do resultado\nTexto explicativo.\nLeitura das métricas\nExplicação dos números.\nPontos de revisão do modelo\nAspectos a revisar.\nLimitações\nLimites da interpretação."}
```

O servidor calcula e envia o tamanho da resposta. A rota web retorna o texto; metadados como modelo retornado e identificador da resposta são registrados pelo fluxo de avaliação, não por esse contrato HTTP. O prompt pede quatro seções, mas a rota não verifica semanticamente se o texto obedece a elas.

A interface e o coletor compartilham a validação da resposta. São aceitos `finish_reason="stop"` e ausência desse metadado, preservando compatibilidade com o contrato anterior. Outros motivos, como `length`, `content_filter` ou `tool_calls`, geram erro; o texto parcial não é exibido como interpretação bem-sucedida. A validação também rejeita conteúdo vazio e a presença da chave configurada no texto ou nos metadados da resposta.

## Erros

| Código | Condição |
|---|---|
| 400 | `Content-Length` ausente ou inválido, JSON inválido ou dados fora do contrato |
| 404 | Rota desconhecida; POST retorna JSON e GET retorna a página HTML de erro do servidor |
| 413 | Corpo acima de 16.384 bytes; comprimento negativo também entra nessa resposta |
| 415 | `POST /interpret` sem `Content-Type: application/json` |
| 502 | Falha HTTP, conexão, timeout, conteúdo vazio, resposta incompleta ou inválida do provedor |
| 503 | Variáveis do provedor ausentes ou URL de configuração inválida |
| 501 | Métodos não implementados pelo servidor, como PUT; resposta HTML da biblioteca padrão |

Exemplo real do formato de validação para um objeto sem os campos exigidos:

```json
{
  "error": "Informe somente diagnosis, probabilities e metrics; outros campos não são aceitos."
}
```

Exemplo de erro de configuração, com status 503:

```json
{
  "error": "Configure LLM_API_URL, LLM_API_KEY e LLM_MODEL no ambiente."
}
```

Uma falha HTTP 401 do provedor chega à API local como status 502, com a mensagem `O provedor LLM respondeu com HTTP 401.`. O corpo original desse erro não é devolvido. Não há interpretação fictícia de fallback ou repetição automática. A chamada HTTPS usa timeout de 30 segundos.

## Saúde e métricas sem chamar a LLM

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/healthz"
Invoke-WebRequest "http://127.0.0.1:8000/metrics" |
    Select-Object -ExpandProperty Content
```

As métricas incluem `interpreter_active_requests`, `interpreter_requests_total`, `interpreter_request_duration_seconds_sum` e `interpreter_request_duration_seconds_count`. Os rótulos de rota são normalizados como `page`, `interpret` ou `other`, acompanhados do status HTTP. Os contadores ficam em memória e reiniciam com o processo. Detalhes em [monitoramento.md](monitoramento.md).

## Limites da demonstração

A aplicação é local, sem autenticação, controle de taxa, limite global de chamadas da interface ou escalabilidade automática. O contrato restringe o conteúdo enviado, mas não transforma o serviço em uma aplicação de produção. A resposta deve ser lida como explicação educativa de saídas de modelo.