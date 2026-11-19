# Tratamento de dados
import pandas as pd
from sklearn.model_selection import train_test_split

# Utilidade
from sklearn.pipeline import make_pipeline

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

def main():
	# ----------------------------------------
	# Lê CSV e separa X e y por colunas
	# ----------------------------------------
	df = pd.read_csv("csv/creditcard_40000.csv")
	
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
	# Modelos
	#	----------------------------------
	models_train = {
		"Decision Tree": DecisionTreeClassifier(),
		"RandomForest": RandomForestClassifier(),
		"SVM": make_pipeline(
			StandardScaler(),
			SVC(
			probability=True)),
		"XGBoost": XGBClassifier()
	}
	
	# ------------------------
	# Treino
	# ------------------------
	for nome, m in models_train.items():
		m.fit(X_train, y_train)
		y_pred = m.predict(X_train)
		y_score = m.predict_proba(X_train)[:, 1]
		roc_auc = roc_auc_score(y_train, y_score)
		pr_auc = average_precision_score(y_train, y_score)
		report = classification_report(y_train, y_pred, target_names=['NORMAL', 'FRAUD'])
		print(f"{nome}:")
		print(report)
		print(f"roc_auc:{roc_auc}")
		print(f"pr_auc:{pr_auc}")
	#	---------------------------
	# Validação
	# ---------------------------
		
	# --------------------------------
	# Teste
	# --------------------------------
	# Métricas
		'''
		cm = confusion_matrix(y_test, y_pred)
		tn, fp, fn, tp = cm.ravel()
		precision = (tn / (tn + fp))
		recalll = (tp / (tp + fn))
		accuracy = accuracy_score(y_test, y_pred)
		f1 = f1_score(y_test, y_pred)
		report = classification_report(y_test, y_pred)
		'''
		
if __name__ == "__main__":
	main()
