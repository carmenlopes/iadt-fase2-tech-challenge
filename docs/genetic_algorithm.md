# Decisões do algoritmo genético

## Genes e domínios

### Regressão Logística

| Gene | Tipo | Domínio | Amostragem/mutação | Restrições |
|---|---|---|---|---|
| `C` | `float` | `[1e-4, 1e4]` | uniforme no expoente base 10 | finito e positivo |
| `solver` | categórico | `lbfgs`, `liblinear`, `saga` | escolha uniforme | determina penalties válidas |
| `penalty` | categórico | `l1`, `l2` | escolha uniforme nas opções do solver | `lbfgs`: `l2`; `liblinear`/`saga`: `l1` ou `l2` |

O pipeline sempre aplica `StandardScaler` dentro do fold. O estimador usa `max_iter=2000`; `ConvergenceWarning` transforma a avaliação em falha explícita. Na versão 1.9 do scikit-learn o parâmetro `penalty` está deprecado, mas ainda é funcional. A versão está fixada e o aviso fica registrado.

### Árvore de Decisão

| Gene | Tipo | Domínio | Amostragem/mutação |
|---|---|---|---|
| `max_depth` | inteiro ou `None` | `None` ou 1 a 20 | escolha uniforme discreta |
| `min_samples_split` | inteiro | 2 a 30 | uniforme discreta |
| `min_samples_leaf` | inteiro | 1 a 15 | uniforme discreta |
| `criterion` | categórico | `gini`, `entropy`, `log_loss` | escolha uniforme |

A árvore não exige padronização; o pipeline contém um passo `passthrough`, mantendo uma interface uniforme e evitando pré-processamento externo.

## Avaliação

Cada indivíduo é validado antes do treino. A avaliação usa `StratifiedKFold` com embaralhamento, seed fixa e cinco folds por padrão. Somente treino entra nessa função.

Por fold são registrados recall, precision, F1, TN, FP, FN e TP. O fitness é a média do recall para `pos_label=1` (maligno); sua dispersão, as demais métricas e os FN também são agregados.

Ordenação determinística dos indivíduos:

1. maior recall médio;
2. maior F1 médio;
3. maior precision média.

A regra não muda durante uma execução. Uma falha ou não convergência não recebe métricas e não pode vencer a seleção.

## Operadores e evolução

- Inicialização independente e válida.
- Torneio sem reposição dentro de cada disputa (`tournament_size=3` por padrão).
- Crossover uniforme: cada gene vem de um dos pais com probabilidade 0,5.
- Mutação independente por gene; genes contínuos/categóricos/inteiros são reamostrados no próprio domínio.
- Reparo posterior garante combinações solver/penalty válidas e limites numéricos.
- Cópias profundas impedem alteração acidental dos pais.
- Elitismo copia os melhores indivíduos para a geração seguinte.
- O número de gerações é o critério de parada; a população mantém tamanho constante.
- Avaliações idênticas são reutilizadas por cache dentro da execução.

## Artefatos por execução

- `config.json`: parâmetros do GA, hash/dimensões do dataset, split, classes e versões do ambiente.
- `history.csv` e `history.json`: melhor fitness, média/desvio da população, melhor indivíduo, complementares, tempo e falhas por geração.
- `best_individual.json`: genes e métricas por fold.
- `summary.json`: objetivo, desempate, confirmação de ausência do teste e falhas detalhadas.
- `evolution.png`: melhor recall e recall médio por geração.

Os diretórios precisam ser novos para impedir sobrescrita acidental de resultados históricos.

