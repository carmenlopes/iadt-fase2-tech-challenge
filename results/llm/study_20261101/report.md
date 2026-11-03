# Avaliação técnica e textual das interpretações

Execução: `study_20261101`. Gerado em 2026-11-02T03:43:32.454591+00:00 (UTC).

Avaliação concluída: **não**.
Meta de qualidade atingida: **não**.

Coleta final: 0/8. Revisão: 0/8.
Respostas satisfatórias: 0/8 previstas; 0/0 revisadas.
Respostas com falha crítica: 0. Falhas de execução: 0.
Tentativas globais: 2/10; nesta execução: 2.

## Protocolo

Dois casos dev e oito finais distintos; uma geração por caso, temperatura 0.2.
Cinco critérios 0–2. Resposta satisfatória: ≥8/10 e nenhuma falha crítica.
Meta interna: ≥90% das oito respostas (8/8) e nenhuma falha crítica.
A conclusão da avaliação independe de alcançar a meta. Notas ausentes são pendências.

## Resultados por critério

| Critério | Média (0–2) |
|---|---|
| fidelity | Pendente |
| statistics | Pendente |
| uncertainty | Pendente |
| utility | Pendente |
| clarity | Pendente |

## Casos e revisão

### final_logistica_original: Regressão logística original

Origem das métricas: project; probabilidades sintéticas.
Pendente: coleta real.

### final_logistica_otimizada: Regressão logística otimizada com baixa confiança

Origem das métricas: project; probabilidades sintéticas.
Pendente: coleta real.

### final_arvore_original: Árvore original com baixa confiança

Origem das métricas: project; probabilidades sintéticas.
Pendente: coleta real.

### final_arvore_otimizada: Árvore otimizada com precisão perfeita no conjunto avaliado

Origem das métricas: project; probabilidades sintéticas.
Pendente: coleta real.

### final_accuracy_enganosa: Accuracy alta e recall baixo

Origem das métricas: synthetic; probabilidades sintéticas.
Pendente: coleta real.

### final_baixa_precisao: Recall alto e precisão baixa

Origem das métricas: synthetic; probabilidades sintéticas.
Pendente: coleta real.

### final_metricas_parciais: Somente recall disponível

Origem das métricas: synthetic; probabilidades sintéticas.
Pendente: coleta real.

### final_extremos: Probabilidade extrema e recall zero

Origem das métricas: synthetic; probabilidades sintéticas.
Pendente: coleta real.

## Falhas de execução

Nenhuma falha registrada; consulte a cobertura antes de inferir sucesso.

## Limitações

- Amostra pequena de oito cenários finais, incluindo dados sintéticos.
- Uma geração por caso; o estudo não mede variabilidade entre gerações da mesma entrada.
- Revisão técnica e textual pela equipe; não houve revisão clínica especializada.
- As notas não constituem validação clínica nem demonstram generalização.

Evidências: manifest.json, cases.json, reviews.json e report.json nesta pasta.
