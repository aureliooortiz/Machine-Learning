#!/usr/bin/env python3

import pandas as pd
import nltk

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import make_scorer, confusion_matrix
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def construir_experimentos():
	# --------------------------------------------------
	# Pipeline Modelos
	# --------------------------------------------------
	pipeline_KNN = Pipeline([
    ('knn', KNeighborsClassifier())
	])

	# --------------------------------------------------
	# Grid
	# --------------------------------------------------
	param_grid_knn = {
		'knn__n_neighbors': [5,9],
		'knn__metric': ['euclidean', 'manhattan', 'cosine']
	}

	return {
		"KNN": (pipeline_knn, param_grid_knn)
	}

def validacao(pipeline, param_grid, X_train, y_train, modelo):
	
	grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', refit='accuracy', n_jobs=-1, verbose=1)
	grid.fit(X_train, y_train)
	
	print("Modelo | Melhores paramêtros | Melhor acurácia média")
	print(f"{modelo}")
	print(f"{grid.best_params_}")
	print(f"{grid.best_score_}")
