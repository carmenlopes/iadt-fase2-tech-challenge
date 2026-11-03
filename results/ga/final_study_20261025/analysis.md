# Análise dos resultados finais

A escolha de hiperparâmetros foi feita exclusivamente pelo recall médio da classe maligna em validação cruzada no treino; F1 e precision foram usados como desempates. O conjunto de teste foi reservado até essa escolha e avaliado uma vez por modelo final.

## Comparação no teste reservado

| Modelo | Recall maligno (baseline → otimizado) | FN (baseline → otimizado) | Accuracy (baseline → otimizado) | Leitura |
|---|---:|---:|---:|---|
| Regressão Logística | 0.9286 → 0.8810 (-0.0476) | 3 → 5 (+2) | 0.9649 → 0.9298 (-0.0351) | Recall diminuiu; FN aumentaram em 2. |
| Árvore de Decisão | 0.8810 → 0.8333 (-0.0476) | 5 → 7 (+2) | 0.9386 → 0.9386 (+0.0000) | Recall diminuiu; FN aumentaram em 2. |

## Interpretação e limites

Recall e falsos negativos devem ser lidos juntos: maior recall significa que uma fração maior dos casos malignos foi identificada, enquanto menos falsos negativos significa menos casos malignos classificados como benignos neste conjunto. Precision e accuracy mostram possíveis custos em falsos positivos e acerto geral. As diferenças observadas valem para este split fixo e não demonstram, isoladamente, desempenho clínico ou generalização para outras populações.

Os resultados dos experimentos e as métricas por geração estão nas pastas de cada execução. A tabela compara o baseline reproduzido da Fase 1 com os candidatos escolhidos sem consultar o teste.
