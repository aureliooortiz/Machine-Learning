# Tratamento de dados
import pandas as pd
from sklearn.model_selection import train_test_split

# Utilidade
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline

# Modelos
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Pré-Processamento
from sklearn.preprocessing import StandardScaler

# Over e Undersampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Class Overlap 
from imblearn.under_sampling import TomekLinks

# Métricas
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	roc_auc_score,
	average_precision_score, 
	confusion_matrix,
	classification_report,
)
from sklearn.metrics import matthews_corrcoef

# Validação
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV 

def main():
	# ----------------------------------------
	# Lê CSV e separa X e y por colunas
	# ----------------------------------------
	df = pd.read_csv("../csv/creditcard_40000.csv")
	
	X = df.drop(columns=["Class"])
	y = df["Class"] # 0 legitima, 1 fraude
	
	# --------------------------------
	# Separa os dados em teste e treino
	# --------------------------------
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)	
	
	# -------------------------------------
	# Separa o treino para validação
	# -------------------------------------
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
	
	# ---------------------------------------------------------------------------
	# Faz base de dados Oversampling e Undersampling, com e sem Overlapping Class
	# ---------------------------------------------------------------------------
	'''	
	tl = TomekLinks()

	X_res_train, y_res_train = tl.fit_resample(X_train, y_train)

	# Amostra undersampling com overlapping class
	und = RandomUnderSampler(random_state=42)

	X_undov_train, y_undov_train = und.fit_resample(X_res_train, y_res_train)

	# Amostra undersampling sem overlapping class
	X_und_train, y_und_train = und.fit_resample(X_train, y_train)

	# Amostra oversampling com overlapping class
	ov = RandomOverSampler(random_state=42)

	X_ovov_train, y_ovov_train = ov.fit_resample(X_res_train, y_res_train)

	# Amostra oversampling sem overlapping class
	X_ov_train, y_ov_train = ov.fit_resample(X_train, y_train)
	'''
	# ----------------------------------
	# Modelos Treino
	#	----------------------------------
	'''
	models_train = {
		"Decision Tree": DecisionTreeClassifier(),
		"RandomForest": RandomForestClassifier(n_jobs=-1, random_state=42),
		"SVM": make_pipeline(
			StandardScaler(),
			SVC()),
		"MLP": make_pipeline( 
			StandardScaler(),
			MLPClassifier(max_iter=500, random_state=42)),	
		"XGBoost": XGBClassifier(n_jobs=-1, random_state=42)
	}
	
	models_train_undov = {
		"Decision Tree": make_pipeline(
			TomekLinks(), 
			RandomUnderSampler(random_state=42),
			DecisionTreeClassifier(),
		),
		"RandomForest": make_pipeline(
			TomekLinks(), 
			RandomUnderSampler(random_state=42),
			RandomForestClassifier(n_jobs=-1, random_state=42),
		),	
		"SVM": make_pipeline(
			StandardScaler(),
			TomekLinks(), 
			RandomUnderSampler(random_state=42),
			SVC(),
		),	
		"MLP": make_pipeline( 
			StandardScaler(),
			TomekLinks(), 
			RandomUnderSampler(random_state=42),
			MLPClassifier(max_iter=500, random_state=42)
		),	
		"XGBoost": make_pipeline(
			TomekLinks(), 
			RandomUnderSampler(random_state=42),
			XGBClassifier(n_jobs=-1, random_state=42),
		),	
	}
	'''
	# ------------------------
	# Treino
	# ------------------------
	'''
	# Treino sem Under nem Oversampling nem Overlapping Class
	for nome, m in models_train.items():
		m.fit(X_train, y_train)
	
	# Treino com Undersampling e Overlapping Class
	for nome, m_undov in models_train.items():
		m_undov.fit(X_undov_train, y_undov_train)
	
	# Treino com Undersampling sem Overlapping Class
	for nome, m_und in models_train.items():
		m_und.fit(X_und_train, y_und_train)
	
	# Treino com Oversampling com Overlapping Class
	for nome, m_ovov in models_train.items():
		m_ovov.fit(X_ovov_train, y_ovov_train)
	
	# Treino com Oversampling sem Overlapping Class
	for nome, m_ov in models_train.items():
		m_ov.fit(X_ov_train, y_ov_train)
	'''
	# -------------------------------------
	# Pipelines
	# -------------------------------------
	'''
	pipelines = {	
		"dt": Pipeline([
			('dt', DecisionTreeClassifier(random_state=42)),
		]),
		
		"rf": Pipeline([
			('rf', RandomForestClassifier(random_state=42))
		]),
		
		"svm": Pipeline([
			('scaler', StandardScaler()),
			('svm', SVC())
		]),
		
		"mlp": Pipeline([
			('scaler', StandardScaler()),
			('mlp', MLPClassifier(max_iter=500, random_state=42))
		]),
		
		"xg": Pipeline([
			('xg', XGBClassifier(n_jobs=1, random_state=42))
		])
	}	
	'''
	# Undersampling + overlapping
	pipelines_undov = {
		"dt": Pipeline([
			('scaler', StandardScaler()),
			('tl', TomekLinks()),
			('und', RandomUnderSampler(random_state=42)),
			('dt', DecisionTreeClassifier(random_state=42)),
		],memory="cache_dir"),
		
		"rf": Pipeline([
			('scaler', StandardScaler()),
			('tl', TomekLinks()),
			('und', RandomUnderSampler(random_state=42)),
			('rf', RandomForestClassifier(random_state=42))
		],memory="cache_dir"),
		
		"svm": Pipeline([
			('scaler', StandardScaler()),
			('tl', TomekLinks()),
			('und', RandomUnderSampler(random_state=42)),
			('svm', SVC())
		],memory="cache_dir"),
		
		"mlp": Pipeline([
			('scaler', StandardScaler()),
			('tl', TomekLinks()),
			('und', RandomUnderSampler(random_state=42)),
			('mlp', MLPClassifier(max_iter=500, random_state=42))
		],memory="cache_dir"),
		
		"xg": Pipeline([
			('scaler', StandardScaler()),
			('tl', TomekLinks()),
			('und', RandomUnderSampler(random_state=42)),
			('xg', XGBClassifier(n_jobs=1, random_state=42))
		],memory="cache_dir")
	}
	
	# Undersampling
	pipelines_und = {
		"dt": Pipeline([
			('und', RandomUnderSampler(random_state=42)),
			('dt', DecisionTreeClassifier(random_state=42)),
		],memory="cache_dir"),
		
		"rf": Pipeline([
			('und', RandomUnderSampler(random_state=42)),
			('rf', RandomForestClassifier(random_state=42))
		],memory="cache_dir"),
		
		"svm": Pipeline([
			('scaler', StandardScaler()),
			('und', RandomUnderSampler(random_state=42)),
			('svm', SVC())
		],memory="cache_dir"),
		
		"mlp": Pipeline([
			('scaler', StandardScaler()),
			('und', RandomUnderSampler(random_state=42)),
			('mlp', MLPClassifier(max_iter=500, random_state=42))
		],memory="cache_dir"),
		
		"xg": Pipeline([
			('und', RandomUnderSampler(random_state=42)),
			('xg', XGBClassifier(n_jobs=1, random_state=42))
		],memory="cache_dir")
	}
	
	# Oversampling + Overlapping
	pipelines_ovov = {
		"dt": Pipeline([
			('scaler', StandardScaler()),
			('tl', TomekLinks()),
			('ov', RandomOverSampler(random_state=42)),
			('dt', DecisionTreeClassifier(random_state=42)),
		],memory="cache_dir"),
		
		"rf": Pipeline([
			('scaler', StandardScaler()),
			('tl', TomekLinks()),
			('ov', RandomOverSampler(random_state=42)),
			('rf', RandomForestClassifier(random_state=42))
		],memory="cache_dir"),
		
		"svm": Pipeline([
			('scaler', StandardScaler()),
			('tl', TomekLinks()),
			('ov', RandomOverSampler(random_state=42)),
			('svm', SVC())
		],memory="cache_dir"),
		
		"mlp": Pipeline([
			('scaler', StandardScaler()),
			('tl', TomekLinks()),
			('ov', RandomOverSampler(random_state=42)),
			('mlp', MLPClassifier(max_iter=500, random_state=42))
		],memory="cache_dir"),
		
		"xg": Pipeline([
			('scaler', StandardScaler()),
			('tl', TomekLinks()),
			('ov', RandomOverSampler(random_state=42)),
			('xg', XGBClassifier(n_jobs=1, random_state=42))
		],memory="cache_dir")
	}

	# Oversampling
	pipelines_ov = {
		"dt": Pipeline([
			('scaler', StandardScaler()),
			('ov', RandomOverSampler(random_state=42)),
			('dt', DecisionTreeClassifier(random_state=42)),
		],memory="cache_dir"),
		
		"rf": Pipeline([
			('scaler', StandardScaler()),
			('ov', RandomOverSampler(random_state=42)),
			('rf', RandomForestClassifier(random_state=42))
		],memory="cache_dir"),
		
		"svm": Pipeline([
			('scaler', StandardScaler()),
			('ov', RandomOverSampler(random_state=42)),
			('svm', SVC())
		],memory="cache_dir"),
		
		"mlp": Pipeline([
			('scaler', StandardScaler()),
			('ov', RandomOverSampler(random_state=42)),
			('mlp', MLPClassifier(max_iter=500, random_state=42))
		],memory="cache_dir"),
		
		"xg": Pipeline([
			('scaler', StandardScaler()),
			('ov', RandomOverSampler(random_state=42)),
			('xg', XGBClassifier(n_jobs=1, random_state=42))
		],memory="cache_dir")
	}
	# ---------------------------------------
	# Hiperparâmetros para validação (Grid)
	# ---------------------------------------	
	grids_val = {	
		"dt": {
			"dt__max_depth": [3,5,10,20,None], 
			"dt__min_samples_split": [2,10,50],
			"dt__max_features":[None, "sqrt"],
			"dt__class_weight": [None, "balanced"],
		},
		
		"rf": {
			"rf__n_estimators":[200,500],
			"rf__max_depth":[10, 20, None], 
			"rf__min_samples_split":[2, 5, 10],
			"rf__bootstrap":[True, False],
			"rf__class_weight": [None, "balanced"],
		},
		
		"svm": [	
			{
			"svm__kernel":["rbf"],
			"svm__C":[0.1, 1, 10, 100],
			"svm__gamma":["scale", 0.1, 0.01, 0.001],
			"svm__class_weight": [None, "balanced"]
			},
			{
			"svm__kernel":["linear"],
			"svm__C":[0.1, 1],
			"svm__class_weight": [None, "balanced"]
			},	
		],
			
		"mlp": {
			"mlp__hidden_layer_sizes": [(50,), (100,),(100, 50), (64,32)],
			"mlp__activation": ["relu", "tanh"],
		},
				
		"xg": {	
			"xg__learning_rate": [0.05, 0.1, 0.3],
			"xg__max_depth": [3,6,9],
			"xg__subsample": [0.8, 1.0],
			"xg__colsample_bytree": [0.8, 1.0],
			"xg__n_estimators": [100, 300],
			"xg__scale_pos_weight": [1, 80],
		}
	}
		
	#	------------------------------------------
	# Validação e Otimização de hiperparametros
	# ------------------------------------------
	
	skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
	# Otimização de hiperparametros sem OS nem US nem OV
	'''
	melhores_train = {}
	for name, pipe in pipelines.items():
		grid = GridSearchCV(pipe, grids_val[name], cv=skf, scoring="average_precision", refit=True, n_jobs=-1, verbose=1)
		grid.fit(X_train, y_train) 
			
		print(f"{name} na Validação:")
		print("Melhores parâmetros:", grid.best_params_)
		print(f"Melhor PR-AUC médio: {grid.best_score_}")

		# Avaliação no conjunto de validação com melhores parametros
		if name != "svm":
			y_val_score = grid.best_estimator_.predict_proba(X_val)[:, 1]
			print(f"PR-AUC:  {average_precision_score(y_val, y_val_score)}")
			print(f"ROC-AUC: {roc_auc_score(y_val, y_val_score)}")
		else:
			y_val_score = grid.best_estimator_.decision_function(X_val)	
			print(f"PR-AUC:  {average_precision_score(y_val, y_val_score)}")
			print(f"ROC-AUC: {roc_auc_score(y_val, y_val_score)}")	
				
		melhores_train[name] = grid.best_estimator_
	'''
	
	melhores_undov = {}
	# Otimização de hiperparametros com US e Ov
	for name, pipe in pipelines_undov.items():
		grid = GridSearchCV(pipe, grids_val[name], cv=skf, scoring="average_precision", refit=True, n_jobs=-1, verbose=1)
		grid.fit(X_train, y_train) 
			
		print(f"{name} na Validação undov:")
		print("Melhores parâmetros:", grid.best_params_)
		print(f"Melhor PR-AUC médio: {grid.best_score_}")

		# Avaliação no conjunto de validação com melhores parametros
		if name != "svm":
			y_val_score = grid.best_estimator_.predict_proba(X_val)[:, 1]
			print(f"PR-AUC:  {average_precision_score(y_val, y_val_score)}")
			print(f"ROC-AUC: {roc_auc_score(y_val, y_val_score)}")
		else:
			y_val_score = grid.best_estimator_.decision_function(X_val)	
			print(f"PR-AUC:  {average_precision_score(y_val, y_val_score)}")
			print(f"ROC-AUC: {roc_auc_score(y_val, y_val_score)}")	
				
		melhores_undov[name] = grid.best_estimator_
	
	melhores_und = {}
	# Otimização de hiperparametros com US sem Ov
	for name, pipe in pipelines_und.items():
		grid = GridSearchCV(pipe, grids_val[name], cv=skf, scoring="average_precision", refit=True, n_jobs=-1, verbose=1)
		grid.fit(X_train, y_train) 
			
		print(f"{name} na Validação und:")
		print("Melhores parâmetros:", grid.best_params_)
		print(f"Melhor PR-AUC médio: {grid.best_score_}")

		# Avaliação no conjunto de validação com melhores parametros
		if name != "svm":
			y_val_score = grid.best_estimator_.predict_proba(X_val)[:, 1]
			print(f"PR-AUC:  {average_precision_score(y_val, y_val_score)}")
			print(f"ROC-AUC: {roc_auc_score(y_val, y_val_score)}")
		else:
			y_val_score = grid.best_estimator_.decision_function(X_val)	
			print(f"PR-AUC:  {average_precision_score(y_val, y_val_score)}")
			print(f"ROC-AUC: {roc_auc_score(y_val, y_val_score)}")	
				
		melhores_und[name] = grid.best_estimator_
	
	melhores_ovov = {}
	# Otimização de hiperparametros com OS e Ov	
	for name, pipe in pipelines_ovov.items():
		grid = GridSearchCV(pipe, grids_val[name], cv=skf, scoring="average_precision", refit=True, n_jobs=-1, verbose=1)
		grid.fit(X_train, y_train) 
			
		print(f"{name} na Validação ovov:")
		print("Melhores parâmetros:", grid.best_params_)
		print(f"Melhor PR-AUC médio: {grid.best_score_}")

		# Avaliação no conjunto de validação com melhores parametros
		if name != "svm":
			y_val_score = grid.best_estimator_.predict_proba(X_val)[:, 1]
			print(f"PR-AUC:  {average_precision_score(y_val, y_val_score)}")
			print(f"ROC-AUC: {roc_auc_score(y_val, y_val_score)}")
		else:
			y_val_score = grid.best_estimator_.decision_function(X_val)	
			print(f"PR-AUC:  {average_precision_score(y_val, y_val_score)}")
			print(f"ROC-AUC: {roc_auc_score(y_val, y_val_score)}")	
				
		melhores_ovov[name] = grid.best_estimator_

	melhores_ov = {}
	# Otimização de hiperparametros com OS sem OV
	for name, pipe in pipelines_ov.items():
		grid = GridSearchCV(pipe, grids_val[name], cv=skf, scoring="average_precision", refit=True, n_jobs=-1, verbose=1)
		grid.fit(X_train, y_train) 
			
		print(f"{name} na Validação ov:")
		print("Melhores parâmetros:", grid.best_params_)
		print(f"Melhor PR-AUC médio: {grid.best_score_}")

		# Avaliação no conjunto de validação com melhores parametros
		if name != "svm":
			y_val_score = grid.best_estimator_.predict_proba(X_val)[:, 1]
			print(f"PR-AUC:  {average_precision_score(y_val, y_val_score)}")
			print(f"ROC-AUC: {roc_auc_score(y_val, y_val_score)}")
		else:
			y_val_score = grid.best_estimator_.decision_function(X_val)	
			print(f"PR-AUC:  {average_precision_score(y_val, y_val_score)}")
			print(f"ROC-AUC: {roc_auc_score(y_val, y_val_score)}")	
				
		melhores_ov[name] = grid.best_estimator_
	
	# --------------------------------
	# Teste
	# --------------------------------
	print("\n\n\n\n\n")
	
	melhores = {
		"undov": melhores_undov,
		"und": melhores_und,
		"ov": melhores_ov,
		"ovov": melhores_ovov
	}
	for tecnica, modelos in melhores.items():

		print(f"\n===== {tecnica} =====")

		for name, modelo in modelos.items():

			y_pred = modelo.predict(X_test)

			print(f"\n{name}")

			report = classification_report(
				y_test, y_pred,
				target_names=['NORMAL', 'FRAUD']
			)
			print(report)

			mat = matthews_corrcoef(y_test, y_pred)
			print(f"Matthews: {mat}")

			if name != "svm":
				y_test_score = modelo.predict_proba(X_test)[:, 1]
			else:
				y_test_score = modelo.decision_function(X_test)

			print(f"PR-AUC: {average_precision_score(y_test, y_test_score)}")
			print(f"ROC-AUC: {roc_auc_score(y_test, y_test_score)}")
			
	'''
	models_test = {	
		"dt": DecisionTreeClassifier(class_weight=None, max_depth=5, max_features='sqrt', min_samples_split=50, random_state=42),
		
		"rf": RandomForestClassifier(bootstrap=False, class_weight='balanced', max_depth=None, min_samples_split=2, n_estimators=500, random_state=42),
		
		"svm": make_pipeline(
			StandardScaler(),
			SVC(C=10, class_weight=None, gamma=0.01, kernel='rbf')
		),
		
		"mlp": make_pipeline( 
			StandardScaler(),
			MLPClassifier(activation='tanh', hidden_layer_sizes=(100,), random_state=42)
		),
		
		"xg": XGBClassifier(colsample_bytree=0.8, learning_rate=0.05, max_depth=6, n_estimators=300, scale_pos_weight=80, subsample=1.0, random_state=42)
	
	}
	
	for name, m in models_test.items():
		m.fit(X_train, y_train)
		y_pred = m.predict(X_test)
		
		print(f"{name}")
		
		report = classification_report(y_test, y_pred, target_names=['NORMAL', 'FRAUD'])
		print(report)
		
		mat = matthews_corrcoef(y_test, y_pred)
		print(f"matthews: {mat}")
		if name != "svm":
			y_test_score = m.predict_proba(X_test)[:, 1]
		else:
			y_test_score = m.decision_function(X_test)	
			
		print(f"PR-AUC:  {average_precision_score(y_test, y_test_score)}")
		print(f"ROC-AUC: {roc_auc_score(y_test, y_test_score)}")	
	'''		
if __name__ == "__main__":
	main()
