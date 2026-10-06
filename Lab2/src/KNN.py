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
		'tfidf__max_features': [350, 500],
		'tfidf__ngram_range': [(1,1)],
		'tfidf__min_df': [5],
		'tfidf__max_df': [0.8, 0.9],
		'knn__n_neighbors': [5,9],
		'knn__metric': ['euclidean', 'manhattan', 'cosine']
	}
	
	param_grid_bow = {
		'bow__max_features': [350, 500],
		'bow__ngram_range': [(1,1)],
		'bow__min_df': [5],
		'bow__max_df': [0.8, 0.9],
		'knn__n_neighbors': [5,9],
		'knn__metric': ['euclidean', 'manhattan', 'cosine']
	}
	
	param_grid_lsa = {
		'tfidf__max_features': [500, 700, 1000],
    'tfidf__ngram_range': [(1,1)],
    'tfidf__min_df': [5],
    'tfidf__max_df': [0.8, 0.9],
    'svd__n_components': [100],
    'knn__n_neighbors': [5,9],
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

def parametros_teste(stop_words_en):
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
		'tfidf__max_features': 500,
		'tfidf__ngram_range': (1,1),
		'tfidf__min_df': 5,
		'tfidf__max_df': 0.8,
		'knn__n_neighbors': 9,
		'knn__metric': 'euclidean'
	}
	
	param_grid_bow = {
		'bow__max_features': 500,
		'bow__ngram_range': (1,1),
		'bow__min_df': 5,
		'bow__max_df': 0.8,
		'knn__n_neighbors': 9,
		'knn__metric': 'cosine'
	}
	
	param_grid_lsa = {
		'tfidf__max_features': 1000,
    'tfidf__ngram_range': (1,1),
    'tfidf__min_df': 5,
    'tfidf__max_df': 0.8,
    'svd__n_components': 100,
    'knn__n_neighbors': 9,
    'knn__metric': 'cosine'
	}
	
	return {
		"TF-IDF": (pipeline_tfidf, param_grid_tfidf),
		"BoW": (pipeline_bow, param_grid_bow),
		"LSA": (pipeline_lsa, param_grid_lsa)
	}

def teste(pipeline, parametros_escolhidos, X_train_texts, y_train, X_test_texts, y_test, nome):
	
	pipeline.set_params(**parametros_escolhidos)
	
	pipeline.fit(X_train_texts, y_train)
	
	y_pred = pipeline.predict(X_test_texts)
	
	acc = accuracy_score(y_test, y_pred)
	prec = precision_score(y_test, y_pred, pos_label=1)
	rec = recall_score(y_test, y_pred, pos_label=1)
	f1 = f1_score(y_test, y_pred, pos_label=1)
	
	tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
	
	print("\n==========================================")
	print(f"      RESULTADOS NO CONJUNTO DE TESTE {nome}    ")
	print("==========================================")
	print("Parâmetros Utilizados:", parametros_escolhidos)
	print(f"  • Acurácia : {acc:.4f}")
	print(f"  • Precisão : {prec:.4f}")
	print(f"  • Recall   : {rec:.4f}")
	print(f"  • F1-Score : {f1:.4f}")
	print("  • Matriz de Confusão:")
	print(f"      VP (Verdadeiros Positivos) : {tp}")
	print(f"      VN (Verdadeiros Negativos) : {tn}")
	print(f"      FP (Falsos Positivos)     : {fp}")
	print(f"      FN (Falsos Negativos)     : {fn}")

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

	X_test_texts = test_df["review"].astype(str)
	y_test = test_df["label"].map({'neg': 0, 'pos': 1})
	
	# --------------------------------------------------
	# Validação
	# --------------------------------------------------
	#experimentos = construir_experimentos(stop_words_en)
	#for nome, (pipeline, grid) in experimentos.items():
		#validacao(pipeline, grid, X_train_texts, y_train, nome)
	
	# --------------------------------------------------
	# Teste
	# --------------------------------------------------
	param_teste = parametros_teste(stop_words_en)
	for nome, (pipeline, parametros) in param_teste.items():
		teste(pipeline, parametros, X_train_texts, y_train, X_test_texts, y_test, nome)
	
if __name__ == "__main__":
    main()


# Dados equilibrados

# Treinamento com dados que vão de 1000 até 20000 exemplos

# Resultado do treinamento na base de teste

