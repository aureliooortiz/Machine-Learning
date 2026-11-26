# Pré-Processamento e facilidade
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

# Percorrer diretorio
from pathlib import Path

# Modelos
from tensorflow import keras
from tensorflow.keras import layers

# Métricas
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Otimização de hiperparametros
import keras_tuner

def normalize(X):
	return scaler.transform(X.reshape(-1, n_mels)).reshape(X.shape)

# Folds é uma lista de ints []
def load_folds(folds):
	X, y = [], []
	for f in folds:
		for arq_pt in sorted((mel_dir / ("fold" + str(f))).glob("*.npy")):
			X.append(np.load(arq_pt))
			# Extrai a classe a partir do nome (.stem)
			name = arq_pt.stem.split("-")
			y.append(int(name[1]))
	return np.stack(X), np.array(y)	# stack estaca exigindo valores no mesmo formato (a, b)	
		
# Define como o modelo aprende		
def compile_model(model, hp):
	model.compile(
		loss="sparse_categorical_crossentropy",
		optimizer=keras.optimizers.Adam(
			learning_rate=hp.Choice("learning_rate", [1e-2, 1e-3, 5e-4, 1e-4])
		),
		metrics=["accuracy"]
	)
	return model

def models_create(input_shape):
	def build_lstm(hp):	
		# Cria modelo sequencial LSTM		
		LSTM = keras.Sequential([
			
			layers.Input(shape=input_shape), # Tipo do input que a rede recebe
			
			layers.LSTM(units=128, return_sequences=True, dropout=hp.Choice("dropout_1", [0.0, 0.2, 0.3, 0.5])), 
			
			layers.LSTM(64, dropout=hp.Choice("dropout_2", [0.0, 0.2, 0.3, 0.5])),
			
			layers.Dense(10, activation="softmax"), # Ultima camada que classifica em probabilidades por softmax
		])
		
		return compile_model(LSTM, hp)
		
	def build_ann(hp):
		ANN = keras.Sequential([
			layers.Input(shape=input_shape),
			layers.Flatten(), 
			layers.Dense(hp.Choice("units_1", [64, 128, 256]), activation="relu"),
			layers.Dropout(hp.Choice("dropout", [0.0, 0.2, 0.5])),
			layers.Dense(hp.Choice("units_2", [32, 64]), activation="relu"),
			layers.Dense(10, activation="softmax"), # 10 classes
		])	
		
		return compile_model(ANN, hp)
	
	def build_cnn(hp):
		filters = hp.Choice("filters", [16, 32, 64])

		CNN = keras.Sequential([
			layers.Input(shape=input_shape), # (tempo, mel)
			layers.Reshape((*input_shape, 1)), # Adiciona o canal: (tempo, mel) -> (tempo, mel, 1)
			layers.Conv2D(filters, 3, activation="relu"),
			layers.MaxPooling2D(2),
			layers.Conv2D(filters * 2, 3, activation="relu"),
			layers.MaxPooling2D(2),
			layers.Flatten(), # Mapas de features -> vetor
			layers.Dropout(hp.Choice("dropout", [0.0, 0.3, 0.5])),
			layers.Dense(10, activation="softmax"), # 10 classes
		])
			
		return compile_model(CNN, hp)	
		
	return {
		"lstm": build_lstm,
		"ann": build_ann,
		"cnn": build_cnn
	}	

def models_test_create(input_shape):
	def build_test_lstm():	
		# Cria modelo sequencial LSTM		
		LSTM = keras.Sequential([
			layers.Input(shape=input_shape), # Tipo do input que a rede recebe
			layers.LSTM(units=128, return_sequences=True, dropout=0.3), 
			layers.LSTM(64, dropout=0.2),
			layers.Dense(10, activation="softmax"), # Ultima camada que classifica em probabilidades por softmax

		])

		LSTM.compile(
			loss="sparse_categorical_crossentropy",
			optimizer=keras.optimizers.Adam(
				learning_rate=5e-4
			),
			metrics=["accuracy"]
		)

		return LSTM

	def build_test_ann():
	
		ANN = keras.Sequential([
			layers.Input(shape=input_shape),
			layers.Flatten(), 
			layers.Dense(units=64, activation="relu"),
			layers.Dropout(rate=0.0),
			layers.Dense(units=64, activation="relu"),
			layers.Dense(10, activation="softmax"), # 10 classes
		])	
				
		ANN.compile(
			loss="sparse_categorical_crossentropy",
			optimizer=keras.optimizers.Adam(
				learning_rate=1e-4
			),
			metrics=["accuracy"]
		)		
				
		return ANN
				
	def build_test_cnn():
		filters = 64
		
		CNN = keras.Sequential([
			layers.Input(shape=input_shape), # (tempo, mel)
			layers.Reshape((*input_shape, 1)), # Adiciona o canal: (tempo, mel) -> (tempo, mel, 1)
			layers.Conv2D(filters, 3, activation="relu"),
			layers.MaxPooling2D(2),
			layers.Conv2D(filters * 2, 3, activation="relu"),
			layers.MaxPooling2D(2),
			layers.Flatten(), # Mapas de features -> vetor
			layers.Dropout(rate=0.0),
			layers.Dense(10, activation="softmax"), # 10 classes
		])

		CNN.compile(
			loss="sparse_categorical_crossentropy",
			optimizer=keras.optimizers.Adam(
				learning_rate=5e-4
			),
			metrics=["accuracy"]
		)
		
		return CNN

	return {
		"lstm": build_test_lstm,
		"cnn": build_test_cnn,
		"ann": build_test_ann
	}

mel_dir = Path("../mel_specs")

# --------------------------------------------
# Separação dos Dados
# --------------------------------------------
'''
X_train, y_train = load_folds(range(1,9)) # 80%
X_val, y_val = load_folds([9]) # 10%
X_test, y_test = load_folds([10]) # 10%

print(X_train.shape, X_val.shape, X_test.shape)
'''
# esperado: (7079, 173, 128) (816, 173, 128) (837, 173, 128)

# --------------------------------------------
# Treino
# ---------------------------------------------
'''
# Normaliza os dados
n_mels = X_train.shape[-1]
scaler = StandardScaler().fit(X_train.reshape(-1, n_mels))

X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)

input_shape = X_train.shape[1:]
builders_train = models_create(input_shape)
'''
''' 
tuners = {}
for name in ["lstm", "ann", "cnn"]:
	tuners[name] = keras_tuner.RandomSearch(
		hypermodel=builders_train[name],
		objective="val_accuracy",
		max_trials=10,
		directory=f"../tuner_{name}",
		project_name=f"{name}_urbansound"
	)
	
	tuners[name].search(
		X_train, y_train,
		epochs=20, batch_size=32,
		validation_data=(X_val, y_val),
		callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)]
	)
	print(name, tuners[name].get_best_hyperparameters(1)[0].values)

best_hps = {}
for name in ["lstm", "ann", "cnn"]:
	best_hps[name] = tuners[name].get_best_hyperparameters(1)[0]
'''
# ------------------------------------
# Teste
# ------------------------------------

classes = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

results = {"lstm": [], "ann": [], "cnn": []}

input_shape = np.load(next((mel_dir / "fold1").glob("*.npy"))).shape   # (173, 128)
builders_test = models_test_create(input_shape)
for name in ["lstm", "ann", "cnn"]:
	for test_fold in range(1,3):
		train_folds = [n for n in range(1,3) if n != test_fold]
		X_train, y_train = load_folds(train_folds)
		X_test, y_test = load_folds([test_fold])
		
		# Normaliza os dados
		n_mels = X_train.shape[-1]
		scaler = StandardScaler().fit(X_train.reshape(-1, n_mels))
		X_train = normalize(X_train)
		X_test = normalize(X_test)
		
		m = builders_test[name]()
		# history guarda curvas de acuracia e perda	
		history = m.fit(
			X_train, y_train,
			epochs=1, batch_size=32,
			validation_data=(X_test, y_test),
			verbose=2
		)
		y_pred = m.predict(X_test).argmax(axis=1)
		
		report = classification_report(y_test, y_pred, labels=range(10), target_names=classes, output_dict=True, zero_division=0)
		cm = confusion_matrix(y_test, y_pred, labels=range(10))
		
		results[name].append({
			"fold": test_fold,
			"acc": report["accuracy"],
			"macro_f1": report["macro avg"]["f1-score"],
			"report": report,
			"cm": cm.tolist(),
			"history": {k: [float(v) for v in vals] for k, vals in history.history.items()}
		})
		
		print(name, "fold", test_fold, "acc", report["accuracy"])
		print(classification_report(y_test, y_pred, labels=range(10), target_names=classes, zero_division=0))
		
		# Salva a cada rodada, para não perder horas de treino se algo cair
		with open(f"../out/results_{name}.json", "w") as f:
			json.dump(results[name], f)
		
		keras.backend.clear_session()
		del X_train, X_test, m 

for name, res in results.items():
	accs = [r["acc"] for r in res]
	print(name, "acc média:", np.mean(accs), "desvio:", np.std(accs))
	print(name, "F1 macro médio:", np.mean([r["macro_f1"] for r in res]))
	
	# Recall por classe (média e desvio entre os folds)
	for c in classes:
		rec = [r["report"][c]["recall"] for r in res]
		print("  ", c, round(np.mean(rec), 3), round(np.std(rec), 3))
	
	# Matriz de confusão somada, normalizada por linha (recall de cada classe)
	cm_total = np.sum([r["cm"] for r in res], axis=0)
	cm_norm = cm_total / cm_total.sum(axis=1, keepdims=True)
	plt.figure(figsize=(8, 7))
	plt.imshow(cm_norm, cmap="Blues")
	plt.xticks(range(10), classes, rotation=90)
	plt.yticks(range(10), classes)
	plt.xlabel("Previsto")
	plt.ylabel("Real")
	plt.colorbar()
	plt.tight_layout()
	plt.savefig(f"../out/cm_{name}.png", dpi=150)
	plt.close()
	
	# Curvas por época (média entre os folds)
	for metrica in ["accuracy", "loss"]:
		treino = np.mean([r["history"][metrica] for r in res], axis=0)
		teste = np.mean([r["history"]["val_" + metrica] for r in res], axis=0)
		plt.figure()
		plt.plot(treino, label="treino")
		plt.plot(teste, label="teste (fold da rodada)")
		plt.xlabel("Época")
		plt.ylabel(metrica)
		plt.legend()
		plt.savefig(f"../out/curva_{metrica}_{name}.png", dpi=150)
		plt.close()

# Comparação entre modelos: acurácia dos 10 folds em um boxplot
plt.boxplot([[r["acc"] for r in results[n]] for n in results], labels=list(results))
plt.ylabel("Acurácia")
plt.savefig("../out/boxplot_modelos.png", dpi=150)
plt.close()

'''
builders_test = models_test_create(input_shape)
for name in ["lstm", "ann", "cnn"]:
	m = builders_test[name]()
	# history guarda curvas de acuracia e perda	
	history = m.fit(
		X_train, y_train,
		epochs=20, batch_size=32
	)
	y_pred = m.predict(X_test).argmax(axis=1)   # Retorna apenas a classificação com base na maior probabilidade
	
	classes = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
	
	print(name)
	report = classification_report(y_test, y_pred, target_names=classes)
	print(report)		
	
lstm.evaluate(X_test, y_test)
y_pred = lstm.predict(X_test).argmax(axis=1)   # Retorna apenas a classificação com base na maior probabilidade

report = classification_report(y_test, y_pred)
print(report)	
'''
