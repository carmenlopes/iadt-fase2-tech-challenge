# Interface de interpretação com LLM

## Arquitetura

O fluxo é local: navegador → servidor Python padrão (`ThreadingHTTPServer`) → validação do payload → endpoint HTTPS compatível com chat completions → interpretação exibida no navegador. O projeto não requer framework web nem SDK adicional.

O serviço `breast_cancer_ga.interpretation` valida os dados antes de montar a solicitação. São aceitos somente:

- classe prevista `benigno` ou `maligno`;
- probabilidades para as duas classes, entre 0 e 1, cuja soma seja aproximadamente 1;
- uma ou mais métricas agregadas entre `accuracy`, `precision`, `recall` e `f1`, cada uma entre 0 e 1.

Campos adicionais são rejeitados. Isso impede que atributos, identificadores ou outros dados sejam encaminhados acidentalmente pelo contrato da interface. O cliente envia esses valores numéricos e a instrução de sistema ao endpoint; não registra o payload em logs.

## Prompt e limites

O prompt de sistema orienta a LLM a:

- responder em português claro, em seções fixas;
- tratar classe e probabilidades como saída de modelo, nunca como diagnóstico confirmado;
- explicar métricas fornecidas sem inventar resultados ou contexto (precision, recall e F1 referem-se à classe maligna neste projeto; accuracy é global);
- apontar um ou dois aspectos acionáveis para a equipe revisar no desempenho/validação do modelo, sem propor condutas para pacientes;
- declarar limitações e incerteza;
- não recomendar tratamento, exames, procedimentos ou condutas para pacientes.

A interface reforça que a explicação é educativa e não substitui avaliação profissional. A interpretação não deve ser usada para tomar decisões de cuidado.

## Configuração e execução

Defina as variáveis no ambiente do processo que iniciará o servidor:

```powershell
$env:LLM_API_URL = "https://seu-provedor/v1/chat/completions"
$env:LLM_API_KEY = "sua-chave"
$env:LLM_MODEL = "nome-do-modelo"
python scripts\serve_interpreter.py
```

`LLM_API_URL` deve ser uma URL HTTPS completa do endpoint de chat completions compatível. O servidor escuta em `127.0.0.1:8000` por padrão; `--host` e `--port` podem ser usados para alterar o endereço de escuta. A URL do endpoint pode incluir um caminho específico do provedor, mas não credenciais ou fragmento. A chave é enviada apenas no cabeçalho de autorização servidor-a-servidor, nunca ao navegador.

Ausência de configuração, entradas inválidas, falhas HTTP, timeout e respostas inválidas do provedor são exibidos como erros; não há fallback que finja uma interpretação bem-sucedida. O conteúdo enviado ao endpoint é externo ao ambiente local: use um provedor cuja política de privacidade seja adequada. Não insira dados pessoais ou identificáveis.

## Testes e avaliação

A especificação das rotas, do contrato JSON, das respostas e dos códigos de erro está em [api.md](api.md). O formulário orienta a informar pelo menos uma métrica e permite deixar as demais em branco. A interface recebe resultados informados; não executa o classificador nem importa automaticamente suas previsões.

A validação da resposta é compartilhada com o coletor. `finish_reason="stop"` ou ausência desse metadado são aceitos. Conteúdo vazio, outros motivos de encerramento e presença da chave configurada no texto ou nos metadados são tratados como erro. A rota web retorna HTTP 502, sem exibir o texto parcial.

Os testes cobrem validação, prompt/contrato de dados, configuração, erros do provedor e rota web sem realizar chamadas externas reais:

```powershell
python -m pytest tests\test_interpretation.py tests\test_web.py
```

Esses testes verificam o comportamento técnico, não a correção clínica ou a qualidade semântica das explicações.