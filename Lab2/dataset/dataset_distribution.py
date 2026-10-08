"""
Histograma + gráfico Q-Q das features de um dataset em formato LIBSVM.

Uso:  python analise_distribuicao.py
Edite a seção CONFIGURAÇÃO abaixo para escolher o arquivo e as features.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_svmlight_file

# ------------------------- CONFIGURAÇÃO -------------------------
ARQUIVO = "train.txt"
N_FEATURES = 132
FEATURES = [1, 13, 22, 101]   # números das features (1 a 132, igual ao arquivo)
CLASSE = None                 # None = todas as classes; ou um número de 0 a 9
IGNORAR_ZEROS = False         # True = analisa só os valores > 0
SALVAR_EM = "distribuicao_features.png"
# ----------------------------------------------------------------


def main():
    X, y = load_svmlight_file(ARQUIVO, n_features=N_FEATURES)
    X, y = X.toarray(), y.astype(int)

    if CLASSE is not None:
        X = X[y == CLASSE]
        print(f"Usando só a classe {CLASSE}: {len(X)} amostras")
    else:
        print(f"Usando todas as classes: {len(X)} amostras")

    fig, eixos = plt.subplots(len(FEATURES), 2,
                              figsize=(11, 3.3 * len(FEATURES)), squeeze=False)

    for linha, f in enumerate(FEATURES):
        todos = X[:, f - 1]                      # coluna da feature (índice 1-based -> 0-based)
        frac_zeros = np.mean(todos == 0)
        v = todos[todos > 0] if IGNORAR_ZEROS else todos

        if len(v) < 20 or v.std() == 0:
            print(f"Feature {f}: dados insuficientes ou constantes, pulando.")
            eixos[linha, 0].set_title(f"Feature {f}: sem dados suficientes")
            continue

        asimetria = stats.skew(v)
        p_norm = stats.normaltest(v).pvalue
        print(f"Feature {f:3d} | zeros={frac_zeros:6.1%} | média={v.mean():.4f} | "
              f"desvio={v.std():.4f} | assimetria={asimetria:6.2f} | p(normal)={p_norm:.2g}")

        # --- Histograma, com a curva normal ajustada por cima para comparar ---
        ax = eixos[linha, 0]
        ax.hist(v, bins=50, density=True, alpha=0.7, edgecolor="white")
        x = np.linspace(v.min(), v.max(), 300)
        ax.plot(x, stats.norm.pdf(x, v.mean(), v.std()), "r--", label="normal ajustada")
        ax.set_title(f"Feature {f}: {frac_zeros:.0%} zeros | assimetria {asimetria:.2f}")
        ax.set_xlabel("valor")
        ax.set_ylabel("densidade")
        ax.legend()

        # --- Q-Q plot: se os pontos seguem a reta, é aproximadamente normal ---
        ax = eixos[linha, 1]
        stats.probplot(v, dist="norm", plot=ax)
        ax.get_lines()[0].set_markersize(3)
        ax.set_title("Gráfico Q-Q (normal)")
        ax.set_xlabel("quantis teóricos")
        ax.set_ylabel("quantis observados")

    titulo = "todas as classes" if CLASSE is None else f"classe {CLASSE}"
    titulo += " | só valores > 0" if IGNORAR_ZEROS else ""
    fig.suptitle(f"Distribuição das features ({titulo})", fontsize=13)
    fig.tight_layout()
    fig.savefig(SALVAR_EM, dpi=130)
    print(f"\nFigura salva em: {SALVAR_EM}")
    plt.show()


if __name__ == "__main__":
    main()
