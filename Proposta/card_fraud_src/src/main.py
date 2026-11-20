# Tratamento de dados
import pandas as pd
from sklearn.model_selection import train_test_split

# Utilidade
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

# Modelos
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Pré-Processamento
from sklearn.preprocessing import StandardScaler

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
	
	# ----------------------------------
	# Modelos Treino
	#	----------------------------------
	models_train = {
		"Decision Tree": DecisionTreeClassifier(),
		"RandomForest": RandomForestClassifier(n_jobs=-1, random_state=42),
		"SVM": make_pipeline(
			StandardScaler(),
			SVC()),
		"XGBoost": XGBClassifier(n_jobs=-1, random_state=42)
	}
	
	# ------------------------
	# Treino
	# ------------------------
	for nome, m in models_train.items():
		m.fit(X_train, y_train)

	# -------------------------------------
	# Pipelines
	# -------------------------------------
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
		
		"xg": Pipeline([
			('xg', XGBClassifier(n_jobs=1, random_state=42))
		])
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
			
		"xg": {	
			"xg__learning_rate": [0.05, 0.1, 0.3],
			"xg__max_depth": [3,6,9],
			"xg__subsample": [0.8, 1.0],
			"xg__colsample_bytree": [0.8, 1.0],
			"xg__n_estimators": [100, 300],
			"xg__scale_pos_weight": [1, 80],
		}
	}
		
	#	---------------------------
	# Validação
	# ---------------------------
	'''
	skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
	melhores = {}
	
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
			y_val_score = grid.best_estimator_.decision_function(X_train)	
			print(f"PR-AUC:  {average_precision_score(y_val, y_val_score)}")
			print(f"ROC-AUC: {roc_auc_score(y_val, y_val_score)}")	
			
		melhores[name] = grid.best_estimator_
	'''	
	# --------------------------------
	# Teste
	# --------------------------------
	models_test = {	
		"dt": DecisionTreeClassifier(class_weight=None, max_depth=5, max_features='sqrt', min_samples_split=50, random_state=42),
		
		"rf": RandomForestClassifier(bootstrap=False, class_weight='balanced', max_depth=None, min_samples_split=2, n_estimators=500, random_state=42),
		
		"svm": make_pipeline(
		StandardScaler(),
		SVC(C=10, class_weight=None, gamma=0.01, kernel='rbf')
		),
		
		"xg": XGBClassifier(colsample_bytree=0.8, learning_rate=0.05, max_depth=6, n_estimators=300, scale_pos_weight=80, subsample=1.0, random_state=42)
	}
	
	for name, m in models_test.items():
		m.fit(X_train, y_train)
		y_pred = m.predict(X_test)
		
		print(f"{name}")
		
		report = classification_report(y_test, y_pred, target_names=['NORMAL', 'FRAUD'])
		print(report)
		
		if name != "svm":
			y_test_score = m.predict_proba(X_test)[:, 1]
			print(f"PR-AUC:  {average_precision_score(y_test, y_test_score)}")
			print(f"ROC-AUC: {roc_auc_score(y_test, y_test_score)}")
		else:
			y_test_score = m.decision_function(X_test)	
			print(f"PR-AUC:  {average_precision_score(y_test, y_test_score)}")
			print(f"ROC-AUC: {roc_auc_score(y_test, y_test_score)}")
			
if __name__ == "__main__":
	main()
