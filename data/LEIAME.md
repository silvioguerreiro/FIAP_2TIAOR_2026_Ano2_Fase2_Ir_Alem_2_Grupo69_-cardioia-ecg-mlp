# Dataset externo (não versionado)

Coloque nesta pasta os dois arquivos do subconjunto PTB do dataset recomendado pelo enunciado:

| Arquivo | Linhas | SHA-256 |
|---|---|---|
| `ptbdb_normal.csv` | 4.046 | `033d355e76e9f65f692fa398c5b24e357d004fab328c97db5160591a2eafe48c` |
| `ptbdb_abnormal.csv` | 10.506 | `27b4a37cd6dbcede0900cde3df02291f21e7538085d54e6b33abb91f4a5cf912` |

Origem: Kaggle, "ECG Heartbeat Categorization Dataset" (https://www.kaggle.com/datasets/shayanfazeli/heartbeat). Baixe o arquivo compactado, extraia apenas os dois CSVs acima para esta pasta e ignore os arquivos `mitbih_*.csv`, que não são usados.

Se os arquivos não estiverem aqui, `src/preparar_imagens.py` e o notebook os obtêm automaticamente de um espelho público e conferem o SHA-256. Os CSVs estão no `.gitignore` (`data/*.csv`) e não devem ser enviados ao GitHub.
