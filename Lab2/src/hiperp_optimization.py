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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
	scoring = {
		'accuracy': 'accuracy',
		'precision': 'precision',
		'recall': 'recall',
		'f1': 'f1',
		'tn': make_scorer(calc_tn),
		'fp': make_scorer(calc_fp),
		'fn': make_scorer(calc_fn),
		'tp': make_scorer(calc_tp)
	}
	
	grid = GridSearchCV(pipeline, param_grid, cv=5, scoring=scoring, refit='accuracy', n_jobs=-1, verbose=1)
	grid.fit(X_train, y_train)
	
	cv_res = grid.cv_results_
	total_combos = len(cv_res['params'])

	print("\n" + "="*80)
	print(f"          RELATÓRIO COMPLETO DE RESULTADOS: {nome_experimento}")
	print("="*80)

	# Imprime os resultados de CADA combinação de parâmetros
	for i in range(total_combos):
		print(f"\n[Combinação {i+1}/{total_combos}]")
		print(f"Parâmetros: {cv_res['params'][i]}")
		print(f"  • Acurácia (média CV) : {cv_res['mean_test_accuracy'][i]:.4f}")
		print(f"  • Precisão (média CV) : {cv_res['mean_test_precision'][i]:.4f}")
		print(f"  • Recall   (média CV) : {cv_res['mean_test_recall'][i]:.4f}")
		print(f"  • F1-Score (média CV) : {cv_res['mean_test_f1'][i]:.4f}")
		print("  • Matriz de Confusão (média por fold):")
		print(f"      VP (Verdadeiros Positivos) : {cv_res['mean_test_tp'][i]:.1f}")
		print(f"      VN (Verdadeiros Negativos) : {cv_res['mean_test_tn'][i]:.1f}")
		print(f"      FP (Falsos Positivos)     : {cv_res['mean_test_fp'][i]:.1f}")
		print(f"      FN (Falsos Negativos)     : {cv_res['mean_test_fn'][i]:.1f}")

	print("\n" + "-"*80)
	print(f">>> VENCEDOR DO {nome_experimento} <<<")
	print("Melhores Parâmetros :", grid.best_params_)
	print(f"Melhor Acurácia     : {grid.best_score_:.4f}")
	print("-"*80 + "\n")
