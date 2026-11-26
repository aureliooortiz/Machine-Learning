# Tratamento de dados (Pre-Processamento)
import librosa 
import soundfile as sf
import torch
import numpy as np
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

# Percorrer diretorio
from pathlib import Path
			
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
		# waveform, sample_rate = torchaudio.load(arq_wav) Da erro
		
		aud, sample_rate = sf.read(arq_wav, dtype="float32")   # mono: shape (n,)
		waveform = torch.from_numpy(aud).unsqueeze(0)           # (1, n), igual ao que o torchaudio devolvia
		
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

		# Troca a extensão .wav por .npy
		final_path = dest_dir / f"{arq_wav.stem}.npy"
		np.save(final_path, mel_db.numpy())
