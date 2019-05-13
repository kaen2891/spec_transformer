import librosa
import numpy as np
import scipy.fftpack
y, sr = librosa.load(librosa.util.example_audio_file(), duration=5, offset=10)
S = np.abs(librosa.stft(y))
mel_spec = librosa.feature.melspectrogram(S=S, sr=sr)
S_inv = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr)
print(np.shape(S_inv))
