import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn import svm

from sklearn.metrics import make_scorer, confusion_matrix

# --------------------------------------------------
# Classes do problema (rótulos 0 a 9)
# --------------------------------------------------
CLASSES = list(range(10))

# --------------------------------------------------
# Função auxiliar para extrair a matriz de confusão
# --------------------------------------------------
# O GridSearchCV só aceita scorers que devolvem UM número. Como a matriz
# de confusão tem 10x10 = 100 células, criamos um scorer por célula (i, j):
# número de exemplos da classe real i que foram previstos como classe j.
def celula_cm(i, j):
	def calc(y_true, y_pred):
		return confusion_matrix(y_true, y_pred, labels=CLASSES)[i, j]
	return calc

def construir_experimentos():
	# --------------------------------------------------
	# Pipeline Modelos
	# --------------------------------------------------
	pipeline_knn = Pipeline([
		('scaler', StandardScaler()),
		('knn', KNeighborsClassifier())
	])
	
	pipeline_nb = Pipeline([
		('gaussianNB', GaussianNB())
	])
	
	pipeline_lda = Pipeline([
		('lda', LinearDiscriminantAnalysis())
	])
	
	pipeline_logistic_regression = Pipeline([
		('scaler', StandardScaler()),
		('logistic_regression', LogisticRegression(max_iter=1000, random_state=0))
	])
	
	pipeline_decision_tree = Pipeline([
		('decision_tree', DecisionTreeClassifier(random_state=0))
	])
	
	pipeline_svm = Pipeline([
		('scaler', StandardScaler()),
		('svm_svc', svm.SVC())
	])

	# --------------------------------------------------
	# Grid
	# --------------------------------------------------
	param_grid_knn = {
		'knn__n_neighbors': [3, 5, 7, 9, 11],
		'knn__weights': ['uniform', 'distance'],
		'knn__metric': ['euclidean', 'manhattan', 'cosine']
	}

	param_grid_nb = {
		'gaussianNB__var_smoothing': [1e-9, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
	}

	# shrinkage não funciona com solver 'svd', por isso a lista de dicionários
	param_grid_lda = [
		{'lda__solver': ['svd']},
		{'lda__solver': ['lsqr', 'eigen'],
		'lda__shrinkage': [None, 'auto', 0.1, 0.5]}
	]

	param_grid_logistic_regression = {
		'logistic_regression__C': [0.01, 0.1, 1, 10, 100]
	}

	param_grid_decision_tree = {
		'decision_tree__criterion': ['gini', 'entropy'],
		'decision_tree__max_depth': [5, 10, 20, None],
		'decision_tree__min_samples_leaf': [1, 5, 20]
	}

	# gamma só existe para kernel rbf, por isso a lista de dicionários
	param_grid_svm = [
		{'svm_svc__kernel': ['rbf'],
		'svm_svc__C': [1, 10, 100],
		'svm_svc__gamma': ['scale', 0.001, 0.01]},
		{'svm_svc__kernel': ['linear'],
		'svm_svc__C': [0.1, 1]}
	]

	return {
		"KNN": (pipeline_knn, param_grid_knn),
		"Naive Bayes": (pipeline_nb, param_grid_nb),
		"LDA": (pipeline_lda, param_grid_lda),
		"Logistic Regression": (pipeline_logistic_regression, param_grid_logistic_regression),
		"Decision Tree": (pipeline_decision_tree, param_grid_decision_tree),
		"SVM": (pipeline_svm, param_grid_svm)
	}

def validacao(pipeline, param_grid, X_train, y_train, modelo):
	# Com mais de 2 classes, 'precision', 'recall' e 'f1' dão erro:
	# é preciso dizer como fazer a média entre as classes (aqui: macro).
	scoring = {
		'accuracy': 'accuracy',
		'precision': 'precision_macro',
		'recall': 'recall_macro',
		'f1': 'f1_macro',
	}
	for i in CLASSES:
		for j in CLASSES:
			scoring[f'cm_{i}_{j}'] = make_scorer(celula_cm(i, j))
	
	grid = GridSearchCV(pipeline, param_grid, cv=5, scoring=scoring, refit='accuracy', n_jobs=-1, verbose=1)
	grid.fit(X_train, y_train)
	
	cv_res = grid.cv_results_
	total_combos = len(cv_res['params'])

	print("\n" + "="*80)
	print(f"          RELATÓRIO COMPLETO DE RESULTADOS: {modelo}")
	print("="*80)

	# Imprime os resultados de CADA combinação de parâmetros
	for k in range(total_combos):
		# Reconstrói a matriz 10x10 (média por fold) a partir dos scorers
		cm = np.array([[cv_res[f'mean_test_cm_{i}_{j}'][k] for j in CLASSES] for i in CLASSES])

		# Por classe (um-contra-todos), derivado da matriz
		vp = np.diag(cm)
		fp = cm.sum(axis=0) - vp
		fn = cm.sum(axis=1) - vp
		vn = cm.sum() - vp - fp - fn

		print(f"\n[Combinação {k+1}/{total_combos}]")
		print(f"Parâmetros: {cv_res['params'][k]}")
		print(f"  • Acurácia (média CV)        : {cv_res['mean_test_accuracy'][k]:.4f}")
		print(f"  • Precisão (média CV, macro) : {cv_res['mean_test_precision'][k]:.4f}")
		print(f"  • Recall   (média CV, macro) : {cv_res['mean_test_recall'][k]:.4f}")
		print(f"  • F1-Score (média CV, macro) : {cv_res['mean_test_f1'][k]:.4f}")

		print("  • Matriz de Confusão (média por fold) - linhas: classe real | colunas: classe prevista")
		print("          " + "".join(f"{j:>7d}" for j in CLASSES))
		for i in CLASSES:
			print(f"      {i:>3d} " + "".join(f"{cm[i, j]:>7.1f}" for j in CLASSES))

		print("  • Por classe (um-contra-todos, média por fold):")
		print("      Classe      VP      VN      FP      FN")
		for c in CLASSES:
			print(f"      {c:>6d} {vp[c]:>7.1f} {vn[c]:>7.1f} {fp[c]:>7.1f} {fn[c]:>7.1f}")

	print("\n" + "-"*80)
	print(f">>> VENCEDOR DO {modelo} <<<")
	print("Melhores Parâmetros :", grid.best_params_)
	print(f"Melhor Acurácia     : {grid.best_score_:.4f}")
	print("-"*80 + "\n")
