# Tech Challenge — Fase 1 - Classificação de Câncer de Mama com Machine Learning

Projeto desenvolvido como parte do **Tech Challenge da pós-graduação IA para Devs**, com foco na aplicação de técnicas de Machine Learning para apoio à análise de dados relacionados ao câncer de mama. Foram utilizados os modelos de **Regressão Logística** e **Árvore de Decisão** para classificação de casos benignos e malignos.

## Dataset Utilizado

Arquivo original obtido no [Kaggle — Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data).

O arquivo .csv utilizado nos notebooks está disponível em dataset-breast-cancer/data.csv

## Tecnologias utilizadas

- Python
- Jupyter Notebook
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jinja2
- SHAP
- numpy

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/carmenlopes/iadt-fase1-tech-challenge.git
```
Entre no diretório:

```bash
cd iadt-fase1-tech-challenge
```

### 2. Crie um ambiente virtual

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

No Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
python -m pip install -r requirements.txt
```

---

## Execução

Com as dependências instaladas, inicie o Jupyter:

```bash
jupyter notebook
```

Abra o notebook desejado e execute as células na ordem apresentada.

Também é possível utilizar os notebooks diretamente pelo VS Code com a extensão Jupyter.

---

## Estrutura do projeto

- `dataset-breast-cancer/` — dataset utilizado no projeto.
- `cancer-regressao-logistica.ipynb` — desenvolvimento e avaliação do modelo de Regressão Logística.
- `cancer-arvore-decisao.ipynb` — desenvolvimento, validação e explicabilidade do modelo de Árvore de Decisão.
- `comparativo.ipynb` — comparação dos resultados e conclusão final.
- `requirements.txt` — dependências necessárias para execução do projeto.
- `img/` - gráficos presentes nos notebooks.