# Modelos
from tensorflow import keras
from tensorflow.keras import layers

# Tratamento de dados (Pre-Processamento)
import librosa 
import soundfile as sf
import torch
import numpy as np
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from sklearn.preprocessing import StandardScaler

# Percorrer diretorio
from pathlib import Path

# Métricas
from sklearn.metrics import classification_report

# Otimização de hiperparametros
import keras_tuner

def normalize(X):
	return scaler.transform(X.reshape(-1, n_mels)).reshape(X.shape)

# Folds é uma lista de ints []
def load_folds(folds):
	X, y = [], []
	for f in folds:
		for arq_pt in sorted((mel_dir / ("fold" + str(f))).glob("*.pt")):
			X.append(torch.load(arq_pt).numpy())
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
			
# ----------------------------------------------
# Pré-Processamento e Leitura de dados de audio
# ----------------------------------------------

# reamostragem do audio -> espectograma de mel + normalização de tamanho + db (0 a -80) + (canal, mel, tempo) = (tempo, mel)

direc = Path("~/.cache/kagglehub/datasets/chrisfilo/urbansound8k/versions/1/").expanduser()

# Diretório onde o script está sendo executado / novo dir
curr_dir = Path.cwd() / "new_wav"

new_wav_dir = Path("../new_wav")

# Faz reamostragem do audio, caso a reamostragem já tenha sido feita não a repete
if not new_wav_dir.exists():
	for arq_wav in sorted(direc.rglob("*.wav")):
		# dir_fold recebe o nome da pasta correspondente do arquivo
		dir_fold = arq_wav.parent.name
		
		# Carrega o arquivo wav e reamostra para 22050 Hz
		aud, sr = librosa.load(arq_wav, sr=22050)
		
		# Cria nova pasta (fold1, fold2, ...)
		dest_dir = curr_dir / arq_wav.parent.name
		dest_dir.mkdir(parents=True, exist_ok=True)
		
		final_path = dest_dir / arq_wav.name
		
		# Salva audio reamostrado
		sf.write(final_path, aud, sr)

# Cada arquivo deve possuir 4 segundos e sr 22050 Hz 
num_samples = 22050 * 4

# n_mels=64 para evitar a criação de bandas vazias
transform = T.MelSpectrogram(sample_rate=22050, n_mels=128, n_fft = 2048, hop_length = 512, f_max=8000)
amplit_transform = T.AmplitudeToDB(top_db=80)

mel_dir = Path("../mel_specs")
# Cria o espectograma de mel com tamanho fixo, normalizados por db, e removido e transposto espec caso esse não exista ainda
if not mel_dir.exists():
	for arq_wav in sorted(new_wav_dir.rglob("*.wav")):
		waveform, sample_rate = torchaudio.load(arq_wav)
		
		# Tamanho fixo: corta se passar de 4s, completa com zeros se faltar 
		# (Pode dar muita bosta)
		waveform = waveform[:, :num_samples]
		lack = num_samples - waveform.shape[1]
		if lack > 0:
			waveform = torch.nn.functional.pad(waveform, (0, lack))
		
		mel_spectrogram = transform(waveform)
		
		# Muda a escala de 0 até -80db
		mel_db = amplit_transform(mel_spectrogram)
		mel_db = mel_db - mel_db.max()
		
		# Remove dimensão canal e inverte tudo
		mel_db = mel_db.squeeze(0)
		mel_db = mel_db.transpose(0,1)
		
		dest_dir = mel_dir / arq_wav.parent.name
		dest_dir.mkdir(parents=True, exist_ok=True)

		# Troca a extensão .wav por .pt
		final_path = dest_dir / f"{arq_wav.stem}.pt"
		torch.save(mel_db, final_path)		

# --------------------------------------------
# Separação dos Dados
# --------------------------------------------
X_train, y_train = load_folds(range(1,9)) # 80%
X_val, y_val = load_folds([9]) # 10%
X_test, y_test = load_folds([10]) # 10%

print(X_train.shape, X_val.shape, X_test.shape)
# esperado: (7079, 173, 128) (816, 173, 128) (837, 173, 128)

# --------------------------------------------
# Treino
# ---------------------------------------------

# Normaliza os dados
n_mels = X_train.shape[-1]
scaler = StandardScaler().fit(X_train.reshape(-1, n_mels))

X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)

builders_train = models_create(X_train.shape[1:])

tuners = {}
for name in ["lstm", "ann", "cnn"]:
	tuners[name] = keras_tuner.RandomSearch(
		hypermodel=builders_train[name],
		objective="val_accuracy",
		max_trials=1,
		directory=f"../tuner_{name}",
		project_name=f"{name}_urbansound"
	)
	
	tuners[name].search(
		X_train, y_train,
		epochs=1, batch_size=32,
		validation_data=(X_val, y_val),
		callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)]
	)
	print(name, tuners[name].get_best_hyperparameters(1)[0].values)
# ------------------------------------
# Teste
# ------------------------------------
# history guarda curvas de acuracia e perda	
'''
lstm.evaluate(X_test, y_test)
y_pred = lstm.predict(X_test).argmax(axis=1)   # Retorna apenas a classificação com base na maior probabilidade

report = classification_report(y_test, y_pred)
print(report)	
'''
