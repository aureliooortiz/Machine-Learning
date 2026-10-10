import time
import numpy as np

from hiperp_optimization import construir_experimentos 
from hiperp_optimization import validacao

from sklearn.datasets import load_svmlight_file 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn import svm

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from plot_matrizes import plotar_matrizes

def main():
	# ------------------------------------------------------
	# Lê arquivo em formato SVM light
	# ------------------------------------------------------
	# Retorna matriz esparsa (csr_matrix) e rótulos
	X_train, y_train = load_svmlight_file("../dataset/train.txt", n_features=132)
	X_test, y_test = load_svmlight_file("../dataset/test.txt", n_features=132)
	
	# --------------------------------------------------------
	# Pré-Processamento
	# ---------------------------------------------------------
	# Converte em matriz densa 
	X_train = X_train.toarray()
	X_test = X_test.toarray()
	#Converte em array
	y_train = y_train.astype(int)
	y_test = y_test.astype(int)
	
	# ---- NOVO: remove as duplicatas do treino ----
	# O train.txt tem 20000 linhas, mas a segunda metade é cópia exata da primeira
	X_train = X_train[:10000]
	y_train = y_train[:10000]
	# ----------------------------------------------
	
	# -----------------------------------------------------------
	# Modelos
	# -----------------------------------------------------------
	models = {
		"KNN": make_pipeline(
			StandardScaler(),
			KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan', n_jobs=-1)
		),		
		"Naive Bayes": GaussianNB(var_smoothing=0.001),
		"LDA": LinearDiscriminantAnalysis(solver='svd'),
		"Logistic Regression": make_pipeline( 
			StandardScaler(),
			LogisticRegression(max_iter=1000, random_state=0, C=0.1)
		),	
		"Decision Tree": DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=20, min_samples_leaf=5),
		"SVM": make_pipeline(
			StandardScaler(),
			svm.SVC(C=10, gamma=0.001, kernel='rbf')
		)	
	}
	
	# -------------------------------------------------
	# Validação
	# --------------------------------------------------
	'''
	experimentos = construir_experimentos()
	for nome, (pipeline, grid) in experimentos.items():
		validacao(pipeline, grid, X_train, y_train, nome)
	'''
	# ------------------------------------------------------------------
	# Treinamento com dados de 1000 em 1000 blocos mantendo a proporção
	# ------------------------------------------------------------------
	print("Modelo número de exemplos: acurácia | Precision | Recall")
	
	cms = {}
	
	erros = {}
	i = 0
	for nome, m in models.items():
		print()
		for n in range(1000, len(X_train) + 1, 1000):
			if n < len(X_train):
				X_sub, _, y_sub, _ = train_test_split(
						X_train, y_train, train_size=n, stratify=y_train, random_state=0
					)
			else:
				X_sub, y_sub = X_train, y_train   # último passo: usa tudo
			
			t0 = time.perf_counter()
			m.fit(X_sub, y_sub)
			t_treino = time.perf_counter() - t0
			
			t0 = time.perf_counter()
			y_pred = m.predict(X_test)
			t_classif = time.perf_counter() - t0
			
			erros[i] = np.where(y_pred != y_test)[0]   # posições dos exemplos errados
			i += 1
			
			acc = accuracy_score(y_test, y_pred)
			if n == len(X_train):	
				cms[nome] = confusion_matrix(y_test, y_pred, labels=range(10))
			
			p = precision_score(y_test, y_pred, average='macro')
			r = recall_score(y_test, y_pred, average='macro')	
			print(f"{nome} em {n} exemplos: {acc:.4f} | precision = {p} | recall = {r}")
	
	np.save("cms.npy", cms)            # guarda, para não precisar retreinar
	plotar_matrizes(cms, "matrizes_confusao.pdf")
	
	'''
	print()
	print("Erros em comum entre cada modelo em cada tamanho de treino")
	for i in range(0, len(erros), 1):
		print()
		for j in range(i+10, len(erros), 10):
			print(len(set(erros[i]) & set(erros[j]))) 
	
	print()
	print("Erros em comum entre todos os modelos usando todos os dados")
	print(len(set(erros[9]) & set(erros[19]) & set(erros[29]) & set(erros[39]) & set(erros[49]) & set(erros[59])))
	'''
if __name__ == "__main__":
    main()


# Dados equilibrados

# Treinamento com dados que vão de 1000 até 20000 exemplos

# Resultado do treinamento na base de teste

