#!/usr/bin/env python3

#import pandas as pd
#import nltk

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminatAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

#from sklearn.metrics import make_scorer, confusion_matrix
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def construir_experimentos():
	# --------------------------------------------------
	# Pipeline Modelos
	# --------------------------------------------------
	pipeline_KNN = Pipeline([
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
		('logistic_regression', LogisticRegression(random_state=0))
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
