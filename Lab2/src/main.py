from sklearn.datasets import load_svmlight_file 
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Funções Auxiliares para Extrair Matriz de Confusão
# --------------------------------------------------
def calc_tn(y_true, y_pred):
	cm = confusion_matrix(y_true, y_pred)
	return cm[0, 0] if cm.shape == (2, 2) else 0

def calc_fp(y_true, y_pred):
	cm = confusion_matrix(y_true, y_pred)
	return cm[0, 1] if cm.shape == (2, 2) else 0

def calc_fn(y_true, y_pred):
	cm = confusion_matrix(y_true, y_pred)
	return cm[1, 0] if cm.shape == (2, 2) else 0

def calc_tp(y_true, y_pred):
	cm = confusion_matrix(y_true, y_pred)
	return cm[1, 1] if cm.shape == (2, 2) else 0

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
	
	# -----------------------------------------------------------
	# Modelos
	# -----------------------------------------------------------
	models = {
		"KNN": KNeighborsClassifier(n_neighbors=9, metric='euclidean')
	}
	
	# -----------------------------------------------------------
	# Treinamento com dados de 1000 em 1000 blocos
	# -----------------------------------------------------------
	for nome, m in models.items():
		m.fit(X_train, y_train)
		y_pred = m.predict(X_test)
		acc = accuracy_score(y_test, m.predict(X_test))
		print(f"{nome}: {acc:.4f}")
	
if __name__ == "__main__":
    main()


# Dados equilibrados

# Treinamento com dados que vão de 1000 até 20000 exemplos

# Resultado do treinamento na base de teste

