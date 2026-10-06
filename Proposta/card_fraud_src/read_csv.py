import pandas as pd

# Nome do arquivo CSV
arquivo = "/home/aurelio/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3/creditcard.csv"

# Lê o arquivo
df = pd.read_csv(arquivo)

# Quantidade total de exemplos
print("Quantidade total de exemplos:", len(df))

# Quantidade de atributos/colunas
print("Quantidade de colunas:", len(df.columns))

# Nomes das colunas
print("\nColunas:")
print(df.columns.tolist())

# Distribuição das classes 
print("\nDistribuição das classes:")
print(df["Class"].value_counts())

# Distribuição percentual
print("\nDistribuição percentual das classes:")
print(df["Class"].value_counts(normalize=True) * 100)

# Valores ausentes
print("\nValores ausentes por coluna:")
print(df.isnull().sum())
