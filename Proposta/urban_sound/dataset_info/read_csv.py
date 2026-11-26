import pandas as pd

# Carrega o dataset
df = pd.read_csv("../csv/UrbanSound8K.csv")

# Mostra a quantidade de exemplos por classe
print("Quantidade de exemplos por classe:")
print(df["class"].value_counts())

# Mostra a porcentagem de cada classe
print("\nPorcentagem de exemplos por classe:")
print(df["class"].value_counts(normalize=True) * 100)

print("\n\n\n")

# Cada gravação (fsID) aparece em um só fold? Deve imprimir 1
print(df.groupby("fsID")["fold"].nunique().max())

# Quantos clipes de cada classe por fold
print(pd.crosstab(df["fold"], df["class"]))

n = df.groupby("fsID")["fold"].nunique()
print((n > 1).sum(), "de", len(n), "gravações em mais de um fold")
print(df[df["fsID"].isin(n[n > 1].index)].sort_values("fsID")[["fsID", "fold", "class"]])

