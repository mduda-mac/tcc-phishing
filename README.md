# tcc-phishing

Detecção de phishing por URL: extração de atributos léxicos e comparação
entre modelos de classificação.

## Dados brutos

Os datasets brutos não são versionados (ver `.gitignore`). Baixe os dois
arquivos e salve-os em `dados/brutos/`:

- **PhiUSIIL Phishing URL Dataset** — https://archive.ics.uci.edu/dataset/967/
  Salvar como `dados/brutos/PhiUSIIL_Phishing_URL_Dataset.csv`.
- **Hannousse e Yahiouche** — DOI [10.17632/c2gw7fy2j4.3](https://doi.org/10.17632/c2gw7fy2j4.3)
  Usar o arquivo `dataset_B_05_2020.csv`, salvo como
  `dados/brutos/dataset_B_05_2020.csv`.

## Ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ordem de execução dos notebooks

1. `notebooks/01_exploracao.ipynb` — exploração dos dados brutos e geração
   das partições em `dados/processados/`.
2. `notebooks/02_modelos.ipynb` — treinamento e avaliação dos modelos.
3. `notebooks/03_figuras.ipynb` — geração das figuras a partir dos
   resultados salvos em `resultados/tabelas/`.
