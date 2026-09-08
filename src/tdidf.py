#!/usr/bin/env python3

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

import pandas as pd
import nltk

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

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
y_train = train_df["label"]

X_test_texts = test_df["review"].astype(str)
y_test = test_df["label"]

#------------------------------------
# Pipeline 
# -----------------------------------
pipeline_tfidf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words_en, lowercase=True)),
    ('knn', KNeighborsClassifier())
])

pipeline_bow = Pipeline([
    ('bow', CountVectorizer(stop_words=stop_words_en, lowercase=True)),
    ('knn', KNeighborsClassifier())
])

pipeline_hash = Pipeline([
    ('hash', HashingVectorizer(stop_words=stop_words_en, lowercase=True, alternate_sign=False)),
    ('knn', KNeighborsClassifier())
])

pipeline_lsa = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words_en, lowercase=True)),
    ('svd', TruncatedSVD(random_state=42)),
    ('knn', KNeighborsClassifier())
])

# -----------------------------------
# Grid 
# -----------------------------------
param_grid_lsa = {
    'tfidf__max_features': [200, 300, 500],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__min_df': [5],
    'tfidf__max_df': [0.8],
    'svd__n_components': [100],
    'knn__n_neighbors': [5],
    'knn__metric': ['cosine']
}
"""
param_grid_hash = {
    'hash__n_features': [200, 300, 500],
    'hash__ngram_range': [(1,1), (1,2)],
    'knn__n_neighbors': [5],
    'knn__metric': ['euclidean', 'manhattan', 'cosine']
}
"""
"""
param_grid_tfidf = {
    'tfidf__max_features': [200, 350, 500],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__min_df': [5, 7, 10],
    'tfidf__max_df': [0.8, 0.9, 1.0],
    'knn__n_neighbors': [5,6,7,8,9,10,11],
    'knn__metric': ['euclidean', 'manhatan', 'cosine']
}
"""
"""
param_grid_tfidf = {
    'tfidf__max_features': [500],
    'tfidf__ngram_range': [(1,1)],
    'tfidf__min_df': [5],
    'tfidf__max_df': [0.8],    
    'knn__weights': ['uniform', 'distance'],
    'knn__n_neighbors': [5],
    'knn__metric': ['cosine']
}
"""
'''
param_grid_bow = {
    'bow__max_features': [200, 350, 500],
    'bow__ngram_range': [(1,1), (1,2)],
    'bow__min_df': [5, 7, 10],
    'bow__max_df': [0.8, 0.9, 1.0],
    'knn__n_neighbors': [5],
    'knn__metric': ['cosine']
}
'''
'''
param_grid_bow = {
        'bow__max_features': [500],
    'bow__ngram_range': [(1,1)],
    'bow__min_df': [5],
    'bow__max_df': [0.8],
    'knn__weights': ['distance'],
    'knn__n_neighbors': [5],
    'knn__metric': ['cosine']
}
'''
# ---------------------------------------
# Roda o grid search em cada pipeline
# ---------------------------------------
"""
grid_tfidf = GridSearchCV(pipeline_tfidf, param_grid_tfidf, cv=5, n_jobs=-1, verbose=1)
grid_tfidf.fit(X_train_texts, y_train)  # passa o TEXTO cru, não o vetorizado!

print("KNN + TFIDF")
print(grid_tfidf.best_params_)
print(grid_tfidf.best_score_)
"""
"""
grid_bow = GridSearchCV(pipeline_bow, param_grid_bow, cv=5, n_jobs=-1, verbose=1)
grid_bow.fit(X_train_texts, y_train)  # passa o TEXTO cru, não o vetorizado!

print("KNN + Bag Of Words")
print(grid_bow.best_params_)
print(grid_bow.best_score_)
"""
'''
grid_hash = GridSearchCV(pipeline_hash, param_grid_hash, cv=5, n_jobs=-1, verbose=1)
grid_hash.fit(X_train_texts, y_train)

print("KNN + Hashing")
print(grid_hash.best_params_)
print(grid_hash.best_score_)
'''

grid_lsa = GridSearchCV(pipeline_lsa, param_grid_lsa, cv=5, n_jobs=-1, verbose=1)
grid_lsa.fit(X_train_texts, y_train)

print("KNN + Hashing")
print(grid_lsa.best_params_)
print(grid_lsa.best_score_)
'''
import pandas as pd
import nltk

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import accuracy_score, classification_report
#from sklearn.model_selection import train_test_split

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
y_train = train_df["label"]

X_test_texts = test_df["review"].astype(str)
y_test = test_df["label"]

# -------------------------------------------------
# Separa os dados d e treino: em treino (80%) e validação (20%)
# -------------------------------------------------

X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    X_train_texts,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

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
# X_val = vectorizer.transform(X_val_texts)
X_test  = vectorizer.transform(X_test_texts)

feature_names = vectorizer.get_feature_names_out()

# --------------------------------------------------
# Validação
# --------------------------------------------------

metric = ["cityblock", "cosine", "euclidean", "haversine", "manhattan", "nan_euclidean"]
for k in range(2,7):
	for m in metric:
		model = KNeighborsClassifier(n_neighbors=k, metric=m)
		model.fit(X_train, y_train) # "treina", armazena os dados de treinamento
		
		predictions = model.predict(X_val) # valida colocando os dados, calculando a distância
		
		acuracia = accuracy_score(y_val, predictions)
		
		print(f"acuracia {k}: {acuracia:.4f}")

param_grid = {
    'n_neighbors': range(2, 7),
    'metric': ['cityblock', 'cosine', 'euclidean', 'haversine', 'manhattan', 'nan_euclidean']
}

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
