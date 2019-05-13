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
    print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))


    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig('Mel-Spectrogram example.png')
    plt.show()
    '''

    return S
# man_original_data = '/mnt/junewoo/workspace/transform/test_wav/man_wav1.wav'
# mel_spec = Mel_S(man_original_data)

def jun_Mel_S(wav_file):
    # mel-spectrogram
    y, sr = librosa.load(wav_file, sr=16000)

    # wav_length = len(y)/sr
    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)
    '''
    print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))


    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig('Mel-Spectrogram example.png')
    plt.show()
    '''

    return S



'''
#for test
filename = '/data/dataset/IEMOCAP_All/IEMOCAP_full_release/Session3/sentences/wav/Ses03M_impro05a/Ses03M_impro05a_F020.wav'

# mel-spectrogram
(_, file_id) = os.path.split(filename)
y, sr = librosa.load(filename, sr=16000)
time = np.linspace(0, len(y)/sr, len(y)) # time axis
fig, ax1 = plt.subplots()
ax1.plot(time, y, color = 'b', label='speech waveform')
ax1.set_xlabel("TIME [s]")
plt.title(file_id)
#plt.plot(time, y)
plt.savefig(file_id+'.png')

plt.show()

#print("length y is : ", len(y))
wav_length = len(y)/sr
#print("wav file length : ", wav_length)

input_nfft = int(round(sr*frame_length))
#print("input nfft type: ",type(input_nfft))
#print("nfft is : ", input_nfft)
input_stride = int(round(sr*frame_stride))
#print("stride is :", input_stride)
#print("input hop_length type: ",type(input_stride))

S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)
#print("S: ", S)
#print("mels shape", np.shape(S))

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title(file_id+' Mel-Spectrogram')
plt.tight_layout()
plt.savefig(file_id+'Mel-Spectrogram.png')
plt.show()
'''


