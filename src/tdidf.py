#!/usr/bin/env python3

import pandas as pd
import nltk

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, confusion_matrix
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

#from sklearn.preprocessing import LabelEncoder

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

def construir_experimentos(stop_words_en):
	# --------------------------------------------------
	# Pipeline
	# --------------------------------------------------
	pipeline_tfidf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words_en, lowercase=True)),
    ('knn', KNeighborsClassifier())
	])

	pipeline_bow = Pipeline([
		  ('bow', CountVectorizer(stop_words=stop_words_en, lowercase=True)),
		  ('knn', KNeighborsClassifier())
	])

	pipeline_lsa = Pipeline([
		  ('tfidf', TfidfVectorizer(stop_words=stop_words_en, lowercase=True)),
		  ('svd', TruncatedSVD(random_state=42)),
		  ('knn', KNeighborsClassifier())
	])
	
	# --------------------------------------------------
	# Grid
	# --------------------------------------------------
	param_grid_tfidf = {
		'tfidf__max_features': [200, 350, 500, 700],
		'tfidf__ngram_range': [(1,1), (1,2)],
		'tfidf__min_df': [5, 7, 10],
		'tfidf__max_df': [0.8, 0.9, 1.0],
		'knn__n_neighbors': [5,7,9,11],
		'knn__metric': ['euclidean', 'manhattan', 'cosine']
	}
	
	param_grid_bow = {
		'bow__max_features': [200, 350, 500, 700],
		'bow__ngram_range': [(1,1), (1,2)],
		'bow__min_df': [5, 7, 10],
		'bow__max_df': [0.8, 0.9, 1.0],
		'knn__n_neighbors': [5,7,9,11],
		'knn__metric': ['euclidean', 'manhattan', 'cosine']
	}
	
	param_grid_lsa = {
		'tfidf__max_features': [200, 350, 500, 700, 1000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__min_df': [5, 7, 10],
    'tfidf__max_df': [0.8, 0.9, 1.0],
    'svd__n_components': [100],
    'knn__n_neighbors': [5,7,9,11],
    'knn__metric': ['euclidean', 'manhattan', 'cosine']
	}
	
	return {
		"TF-IDF": (pipeline_tfidf, param_grid_tfidf),
		"BoW": (pipeline_bow, param_grid_bow),
		"LSA": (pipeline_lsa, param_grid_lsa)
	}

def validacao(pipeline, param_grid, X_train_texts, y_train, nome_experimento):
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
	grid.fit(X_train_texts, y_train)
	
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

def main():
	# --------------------------------------------------
	# Load NLTK stopwords
	# --------------------------------------------------
	try:
		stop_words_en = stopwords.words('english')
	except LookupError:
		nltk.download('stopwords')
		stop_words_en = stopwords.words('english')
		
	# --------------------------------------------------
	# Read CSVs
	# --------------------------------------------------
	train_df = pd.read_csv("../txt/comments_train.txt")
	test_df = pd.read_csv("../txt/comments_test.txt")
	
	# --------------------------------------------------
	# Dados
	# --------------------------------------------------
	X_train_texts = train_df["review"].astype(str)
	y_train = train_df["label"].map({'neg': 0, 'pos': 1})
	#y_train = train_df["label"]

	X_test_texts = test_df["review"].astype(str)
	y_test = test_df["label"].map({'neg': 0, 'pos': 1})
	#y_test = test_df["label"]

	# --------------------------------------------------
	# Validação
	# --------------------------------------------------
	experimentos = construir_experimentos(stop_words_en)
	for nome, (pipeline, grid) in experimentos.items():
		validacao(pipeline, grid, X_train_texts, y_train, nome)
				
if __name__ == "__main__":
    main()

"""
grid_tfidf = GridSearchCV(pipeline_tfidf, param_grid_tfidf, cv=5, n_jobs=-1, verbose=1)
grid_tfidf.fit(X_train_texts, y_train)  # passa o TEXTO cru, não o vetorizado!

print("KNN + TFIDF")
print(grid_tfidf.best_params_)
print(grid_tfidf.best_score_)

grid_bow = GridSearchCV(pipeline_bow, param_grid_bow, cv=5, n_jobs=-1, verbose=1)
grid_bow.fit(X_train_texts, y_train)  # passa o TEXTO cru, não o vetorizado!

print("KNN + Bag Of Words")
print(grid_bow.best_params_)
print(grid_bow.best_score_)

grid_hash = GridSearchCV(pipeline_hash, param_grid_hash, cv=5, n_jobs=-1, verbose=1)
grid_hash.fit(X_train_texts, y_train)

print("KNN + Hashing")
print(grid_hash.best_params_)
print(grid_hash.best_score_)

grid_lsa = GridSearchCV(pipeline_lsa, param_grid_lsa, cv=5, n_jobs=-1, verbose=1)
grid_lsa.fit(X_train_texts, y_train)

print("KNN + LSA")
print(grid_lsa.best_params_)
print(grid_lsa.best_score_)
"""
'''
# --------------------------------------------------
# TF-IDF
# --------------------------------------------------
print ("Extraindo representacao...")
vectorizer = TfidfVectorizer(
	stop_words=stop_words_en,
	max_features=350,
	ngram_range=(1,2),
	lowercase=True,
	min_df=2, max_df=.9
)

X_train = vectorizer.fit_transform(X_train_texts)
X_test  = vectorizer.transform(X_test_texts)

feature_names = vectorizer.get_feature_names_out()

# --------------------------------------------------
# Validação
# --------------------------------------------------
# Converte os rótulos em números:
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print("Testando hiperparâmetros...")
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)

# --------------------------------------------------
# Modelo
# --------------------------------------------------

print ("Classificando com kNN...")

model = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
model.fit(X_train, y_train) # "treina", armazena os dados de treinamento

# --------------------------------------------------
# Predição
# --------------------------------------------------
predictions = model.predict(X_test) # testa colocando os dados, calculando a distância
#probs = model.predict_proba(X_test)

# --------------------------------------------------
# Avaliação
# --------------------------------------------------
from sklearn.metrics import classification_report
print("\nEvaluation:")
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions) # Compara os rótulos com a predição feita
print (cm)
'''
