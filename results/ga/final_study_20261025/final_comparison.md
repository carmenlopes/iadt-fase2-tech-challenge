# Comparação final: baseline e modelo otimizado

Os candidatos otimizados foram selecionados por validação cruzada somente no treino, entre três configurações de GA por modelo. O teste reservado foi usado depois da seleção para a avaliação final.

| Modelo | Versão | Accuracy | Precision maligna | Recall maligno | F1 maligno | FN | Matriz [[TN, FP], [FN, TP]] |
|---|---|---:|---:|---:|---:|---:|---|
| Regressão Logística | Original | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 3 | `[[71, 1], [3, 39]]` |
| Regressão Logística | Otimizado | 0.9298 | 0.9250 | 0.8810 | 0.9024 | 5 | `[[69, 3], [5, 37]]` |
| Árvore de Decisão | Original | 0.9386 | 0.9487 | 0.8810 | 0.9136 | 5 | `[[70, 2], [5, 37]]` |
| Árvore de Decisão | Otimizado | 0.9386 | 1.0000 | 0.8333 | 0.9091 | 7 | `[[72, 0], [7, 35]]` |

## Hiperparâmetros escolhidos

- **Regressão Logística** — experimento `experiment_03_logistic_regression`; genes: `{"C": 3507.0604419517017, "penalty": "l2", "solver": "lbfgs"}`; Recall CV=0.9647, F1 CV=0.9593, Precision CV=0.9552.
- **Árvore de Decisão** — experimento `experiment_02_decision_tree`; genes: `{"criterion": "log_loss", "max_depth": 20, "min_samples_leaf": 14, "min_samples_split": 19}`; Recall CV=0.9412, F1 CV=0.9305, Precision CV=0.9207.

Métricas originais vêm da reprodução da Fase 1 em `results/phase1_baseline/baseline_metrics.json`.
