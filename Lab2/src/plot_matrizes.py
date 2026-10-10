import numpy as np
import matplotlib.pyplot as plt

# Fontes TrueType (tipo 42) no PDF: evita fontes Type 3, que algumas
# conferências sinalizam na verificação do PDF.
plt.rcParams["pdf.fonttype"] = 42


def plotar_matrizes(cms, arquivo="matrizes_confusao.pdf", figsize=(7.0, 3.8)):
	"""
	cms: dicionário {nome_do_modelo: matriz de confusão (10x10)}.

	Cada linha é normalizada pela classe real (fração de cada classe que foi
	prevista como cada uma das outras). A diagonal (acertos) fica em cinza,
	para que a escala de cor mostre só os ERROS, com a mesma escala em todos
	os modelos, o que permite comparar um modelo com o outro.
	"""
	erro = {}
	for nome, cm in cms.items():
		cmn = cm / cm.sum(axis=1, keepdims=True)
		np.fill_diagonal(cmn, np.nan)
		erro[nome] = cmn
	vmax = np.nanmax([np.nanmax(e) for e in erro.values()])

	cmap = plt.get_cmap("Reds").copy()
	cmap.set_bad("#e6e6e6")   # cor da diagonal

	fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
	for ax, (nome, e) in zip(axes.ravel(), erro.items()):
		im = ax.imshow(e, cmap=cmap, vmin=0, vmax=vmax)
		ax.set_title(nome, fontsize=8)
		ax.set_xticks(range(10))
		ax.set_yticks(range(10))
		ax.tick_params(labelsize=5.5, length=2)

	cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
	cb.ax.tick_params(labelsize=6)
	cb.set_label("Fração da classe real", fontsize=7)
	fig.supxlabel("Classe prevista", fontsize=7)
	fig.supylabel("Classe real", fontsize=7)
	fig.savefig(arquivo)
	plt.close(fig)


if __name__ == "__main__":
	# Uso independente: carrega as matrizes salvas pelo main.py
	cms = np.load("cms.npy", allow_pickle=True).item()
	plotar_matrizes(cms)
