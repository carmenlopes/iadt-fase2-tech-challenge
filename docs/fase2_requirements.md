# Requisitos oficiais da Fase 2 e recorte desta execução

Fonte principal: `IADT - Fase 2 - Tech challenge.pdf`, páginas 2 a 7. O projeto escolhido é o **Projeto 1 - Otimização de Modelos de Diagnóstico**.

## Mapeamento

| Requisito do PDF | Natureza | Situação nesta etapa |
|---|---|---|
| Codificar hiperparâmetros como genes | Obrigatório | Concluído para regressão e árvore |
| Implementar seleção, cruzamento e mutação | Obrigatório | Concluído; torneio, crossover uniforme e mutação por gene |
| Fitness baseada em métricas de desempenho | Obrigatório | Concluído; recall maligno principal, F1 e precision como desempate |
| Comparar modelos otimizados e originais | Obrigatório da entrega final | Pendente; teste reservado não foi usado nesta etapa |
| Ao menos 3 experimentos com configurações de GA distintas | Obrigatório da entrega final | Pendente; as duas buscas atuais são verificações funcionais |
| Monitoramento e logging de desempenho | Obrigatório | Base concluída para a busca: configuração, geração, tempo, métricas e falhas |
| Documentar arquitetura e decisões | Obrigatório | Base concluída no README e documentação técnica |
| Integração com LLM, prompt engineering e avaliação | Obrigatório da entrega final | Pendente e fora do recorte até 11/10/2026 |
| Projeto Python estruturado e ambiente virtual | Obrigatório para ambos os projetos | Concluído |
| Testes automatizados | Obrigatório para ambos os projetos | Concluído para o escopo atual |
| Código, documentação e scripts/notebooks de demonstração | Entregável | Base concluída; relatório final ainda pendente |
| Relatório técnico | Entregável | Pendente |
| Vídeo de demonstração de até 15 minutos | Entregável | Pendente |
| Nuvem e IaC | Opcional/pontuação extra | Não implementado |

## Divergências e decisões de escopo

O PDF cita “accuracy, recall, F1-score, etc.” sem prescrever uma única função objetivo. Esta implementação usa recall da classe maligna como fitness principal porque falsos negativos têm maior custo no contexto estudado. F1 e precision são desempates declarados e métricas de controle.

O PDF pede comparação final entre originais e otimizados. Ela foi adiada deliberadamente: as buscas desta etapa usam apenas CV no treino, e a avaliação final no teste deve ocorrer uma única vez depois que o protocolo dos três experimentos estiver fechado.

“Configurar recursos de escalabilidade automática” aparece como obrigatório, mas a observação seguinte declara nuvem opcional. Neste recorte foi implementado logging local reproduzível. Escalabilidade de infraestrutura, implantação e IaC permanecem pendentes/opcionais conforme a decisão futura do grupo.

## Evidências da Fase 1

- Dataset Kaggle com 569 linhas e 30 features após remoção de `id` e coluna vazia.
- Mapeamento verificável nos notebooks: `M -> 1`, `B -> 0`.
- Split estratificado de 20% para teste, `random_state=42`.
- Regressão: `StandardScaler` ajustado no treino; defaults de `LogisticRegression`.
- Árvore final: `max_depth=7`, `random_state=42`, escolhido por `StratifiedKFold(5, shuffle=True, random_state=42)` no treino.
- Não foi identificado vazamento entre treino e teste no procedimento reproduzido.

