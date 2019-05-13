# -*- coding: utf-8 -*-
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

frame_length = 0.025
frame_stride = 0.010

def Mel_S(wav_file, fs):
    # mel-spectrogram
    y, sr = librosa.load(wav_file, sr=fs)

    # wav_length = len(y)/sr
    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))
    print("wav length is ", len(y)/sr)
    S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)
    '''
    # plot figure
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig('Mel-Spectrogram example.png')
    plt.show()
    '''

    return S