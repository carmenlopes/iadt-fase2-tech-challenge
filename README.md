# Tech Challenge — Fase 2: otimização de modelos de câncer de mama

Continuação do projeto da Fase 01, com algoritmo genético para buscar hiperparâmetros de Regressão Logística e Árvore de Decisão e uma interface local para interpretação de resultados com LLM. A metodologia, os experimentos e a análise dos resultados estão no relatório técnico da Fase 2.

## O que é da Fase 01

A Fase 01 contém o estudo inicial de classificação de tumores como benignos ou malignos, com preparação dos dados, treinamento e comparação dos dois modelos. Seus artefatos foram preservados:

- [01fase/](01fase/): notebooks de Regressão Logística, Árvore de Decisão e comparação dos modelos.
- [dataset-breast-cancer/data.csv](dataset-breast-cancer/data.csv): conjunto de dados utilizado nos estudos.
- [01fase/img/](01fase/img/): figuras do estudo original.

Esses modelos são a referência (baseline) para a otimização da Fase 2.

## Instalação e execução

Execute os comandos na raiz do repositório, pelo Windows PowerShell. O projeto declara Python 3.11 ou superior; as dependências fixadas foram usadas com Python 3.14.

```powershell
py -3.14 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

Com o ambiente virtual ativo, os principais comandos são:

| Objetivo | Comando |
|---|---|
| Abrir os notebooks da Fase 01 | `jupyter notebook 01fase` |
| Reproduzir os modelos de referência | `python scripts\reproduce_baseline.py` |
| Executar os testes automatizados | `python -m pytest` |
| Executar o estudo completo, com três buscas por modelo | `python scripts\run_final_study.py --study-id meu_estudo` |
| Iniciar a interface e a API local | `python scripts\serve_interpreter.py` |

Para executar uma busca pequena por modelo:

```powershell
python scripts\run_ga.py --model logistic_regression --run-id minha_logistica --population-size 8 --generations 3 --folds 5 --seed 42
python scripts\run_ga.py --model decision_tree --run-id minha_arvore --population-size 8 --generations 3 --folds 5 --seed 42
```

Os resultados são gravados em `results/`. Use identificadores novos em `--run-id` e `--study-id` para cada execução. Para consultar os parâmetros disponíveis, acrescente `--help` ao comando do script.

## Interface de interpretação com LLM

A interface recebe uma classe prevista, as probabilidades de benigno e maligno e uma ou mais métricas agregadas (`accuracy`, `precision`, `recall` e `f1`). Com esses valores, solicita a um provedor LLM uma explicação educativa em português. A previsão deve ser informada pelo usuário; o formulário não executa os modelos de classificação.

Configure as variáveis de ambiente e inicie o servidor:

```powershell
$env:LLM_API_URL = "https://seu-provedor/v1/chat/completions"
$env:LLM_API_KEY = "sua-chave"
$env:LLM_MODEL = "nome-do-modelo"
python scripts\serve_interpreter.py
```

Abra **http://127.0.0.1:8000** e preencha o formulário. Use `Ctrl+C` no terminal para encerrar. O servidor aceita `--host` e `--port` e, por padrão, escuta apenas no computador local.

O provedor deve oferecer um endpoint HTTPS compatível com chat completions. A chave permanece no servidor, e os dados enviados se limitam à classe, às probabilidades e às métricas aceitas. Não informe dados pessoais ou atributos de pacientes. O prompt pede uma explicação com limitações, sem confirmação de diagnóstico ou recomendações de tratamento; a aplicação é uma demonstração acadêmica.

Detalhes de configuração em [docs/llm_interpretation.md](docs/llm_interpretation.md) e comandos de avaliação das respostas em [docs/llm_evaluation.md](docs/llm_evaluation.md).

## API local e suas três rotas principais

A API usa `ThreadingHTTPServer`, da biblioteca padrão do Python, e inicia junto com a interface.

| Método e rota | Finalidade | Resposta de sucesso |
|---|---|---|
| `GET /` | Exibir o formulário web | Página HTML |
| `POST /interpret` | Validar os valores informados e solicitar a explicação à LLM | JSON com o campo `interpretation` |
| `GET /healthz` | Verificar se o servidor local responde | `{"status":"ok"}` |

A rota de saúde não verifica a disponibilidade do provedor LLM. Há também a rota auxiliar `GET /metrics`, que expõe contadores de requisições e duração para monitoramento.

Exemplo de chamada pelo PowerShell, com o servidor configurado e em execução:

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

O exemplo usa valores sintéticos e faz uma chamada real ao provedor, sujeita à cobrança dele. O contrato completo, as validações e os códigos de erro estão em [docs/api.md](docs/api.md). As métricas estão descritas em [docs/monitoramento.md](docs/monitoramento.md).

## Estrutura adicionada e design patterns utilizados

```text
src/breast_cancer_ga/
    data.py             carregamento e divisão dos dados
    models.py           genes, validação e construção dos modelos
    fitness.py          avaliação dos candidatos por validação cruzada
    ga.py               configuração e execução do algoritmo genético
    reporting.py        gravação de métricas, históricos e gráficos
    interpretation.py   validação dos dados, prompt e integração com LLM
    web.py              formulário, rotas HTTP e métricas do serviço
    evaluation.py       coleta e avaliação das respostas da LLM
scripts/                comandos de reprodução, buscas, servidor e avaliação
tests/                  testes automatizados
evaluation/             casos versionados para avaliação da LLM
results/                artefatos gerados pelas execuções
docs/                   documentação de apoio
```

Os padrões e práticas aparecem de forma simples, por funções e composição:

| Padrão ou prática | Onde aparece e para que serve |
|---|---|
| **Strategy com injeção de dependência** | `GeneticAlgorithm` recebe `fitness_function`, permitindo trocar a função de avaliação sem alterar a evolução. `create_server` também recebe uma função `interpreter`, substituível nos testes. |
| **Fábrica simples (Simple Factory)** | `build_pipeline`, em `models.py`, centraliza a criação do pipeline de Regressão Logística ou Árvore de Decisão. |
| **Composição com Pipeline** | O `Pipeline` do scikit-learn reúne pré-processamento e modelo em um único fluxo de treinamento e previsão. |
| **Objetos de dados (DTOs)** | Dataclasses como `GAConfig`, `DatasetSplit`, `FitnessResult` e `InterpretationInput` agrupam configurações, entradas e resultados com campos explícitos. |
| **Memoização** | O cache de `GeneticAlgorithm` reutiliza a avaliação de indivíduos com os mesmos genes durante uma busca, evitando treinamentos repetidos. |
