import pandas as pd

# Arquivo original
arquivo = "/home/aurelio/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3/creditcard.csv"

# Lê o dataset
df = pd.read_csv(arquivo)

# Quantidade desejada de exemplos
n_classe_0 = 39508

# Seleciona todos os exemplos de fraude
classe_1 = df[df["Class"] == 1]

# Seleciona 39.508 exemplos normais
classe_0 = df[df["Class"] == 0].sample(
    n=n_classe_0,
    random_state=42
)

# Junta as duas classes
df_reduzido = pd.concat([classe_0, classe_1])

# Embaralha os exemplos
df_reduzido = df_reduzido.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Salva o novo dataset
df_reduzido.to_csv("creditcard_40000.csv", index=False)

# Mostra informações
print("Quantidade total:", len(df_reduzido))

print("\nDistribuição das classes:")
print(df_reduzido["Class"].value_counts())

print("\nDistribuição percentual:")
print(df_reduzido["Class"].value_counts(normalize=True) * 100)
