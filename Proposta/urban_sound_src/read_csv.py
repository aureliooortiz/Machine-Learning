import pandas as pd

# Carrega o dataset
df = pd.read_csv("/home/aurelio/.cache/kagglehub/datasets/chrisfilo/urbansound8k/versions/1/UrbanSound8K.csv")

# Mostra a quantidade de exemplos por classe
print("Quantidade de exemplos por classe:")
print(df["class"].value_counts())

# Mostra a porcentagem de cada classe
print("\nPorcentagem de exemplos por classe:")
print(df["class"].value_counts(normalize=True) * 100)
