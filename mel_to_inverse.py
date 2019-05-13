import librosa
import numpy as np
import scipy.fftpack
import soundfile as sf
import extract_Mel_Spectrogram
import os
import matplotlib.pyplot as plt

class Convert2Wav:
    ''' Convert to Wav module '''

    def __init__(self, wav, mel_spec, frame_length, frame_stride, sr, save_dir, save_name):
        super().__init__()

        self.wav = wav
        self.mel_spec = mel_spec
        self.frame_length = frame_length
        self.frame_stride = frame_stride
        # self.input_nfft = input_nfft
        # self.input_stride = input_stride
        self.sr = sr
        self.save_dir = save_dir
        self.save_name = save_name

        self.mel_nfft = int(round(self.sr * self.frame_length))
        self.mel_hoplength = int(round(self.sr * self.frame_stride))

    def mel2wav(self):
        stft = librosa.feature.inverse.mel_to_stft(self.mel_spec, n_fft=self.mel_nfft, sr=self.sr)

        # method griffin
        spec_inverse = librosa.core.griffinlim(stft, n_iter=32, hop_length=self.mel_hoplength, window='hann')

        # method mel_to_audio
        final_dir = os.path.join(self.save_dir, self.save_name)
        sf.write(final_dir+'_inverse.wav', spec_inverse, self.sr, format='WAV', endian='LITTLE', subtype='PCM_16')
        print("saved name {} finished ".format(final_dir))

    # def wav2mel(self):
    #     Mel_Spectrogram = extract_Mel_Spectrogram.Mel_S(self.wav, self.sr, self.frame_length, self.frame_stride)
    #     print('Mel shape is :', np.shape(Mel_Spectrogram))
    #     return Mel_Spectrogram

    '''
    def spec2wav(self):
        # S_hat = librosa.istft(S)
        # sf.write('./test_wav/' + save_name + 'S_stft_istft.wav', S_hat, 16000, format='WAV', endian='LITTLE',
        #          subtype='PCM_16')
        S_inv_hat = librosa.istft(self.mel_spec)
        print("save dir is :", self.save_dir)
        sf.write(self.save_dir +'S_inv_istft.wav', S_inv_hat, 16000, format='WAV', endian='LITTLE',
                 subtype='PCM_16')
    '''


'''
frame_length = 0.025
frame_stride = 0.010
input_nfft = int(round(16000*frame_length))
input_stride = int(round(16000*frame_stride))

def mel2flot(wav):
    y, sr = librosa.load(wav, sr=16000)
    original_S = librosa.stft(y)
    S = np.abs(librosa.stft(y))
    mel_spec = librosa.feature.melspectrogram(S=S, sr=sr)
    S_inv = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr)

    # plt.figure()
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max, top_db=None), y_axis = 'log', x_axis='time')
    plt.colorbar()
    plt.title('Original STFT')
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(S_inv - S), ref=S.max(), top_db=None), vmax=0, y_axis='log', x_axis='time', cmap='magma')
    plt.title('Residual error (dB)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('result of stft, mel')
    return original_S, S_inv



# S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)

def wav2mel(wav_dir):
    #y, sr = librosa.load(wav_dir, sr=16000)
    Mel_Spectrogram = extract_Mel_Spectrogram.jun_Mel_S(wav_dir)
    print('Mel shape is :', np.shape(Mel_Spectrogram))
    return Mel_Spectrogram

def mel2stft(mel_spec, sex, save_dir):
    # stft = mel_to_stft(M, sr=sr, n_fft=n_fft, power=power, **kwargs)
    stft = librosa.feature.inverse.mel_to_stft(mel_spec, n_fft=input_nfft, sr=16000)

    #method griffin
    audio = librosa.core.griffinlim(stft, n_iter=32, hop_length=input_stride, window='hann')

    #method mel_to_audio
    audio2 = librosa.feature.inverse.mel_to_audio(stft)
    # audio = librosa.core.griffinlim(stft)
    final_dir = os.path.join(save_dir, sex)

    sf.write(final_dir+'1.wav', audio, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
    # sf.write(final_dir + '2.wav', audio, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
    print("save finish")
# sr = 16000
man_wav = '/mnt/junewoo/workspace/transform/test_wav/man_wav1.wav'
woman_wav = '/mnt/junewoo/workspace/transform/test_wav/woman_wav1.wav'
save_dir =  '/mnt/junewoo/workspace/transform/test_wav/'
#man
S, S_inv = mel2flot(man_wav)
save_name = 'man'
spec2wav(S, S_inv, save_name)

#woman
S2, S_inv2 = mel2flot(woman_wav)
save_name = 'woman'
spec2wav(S2, S_inv2, save_name)

mel_spec_man = wav2mel(man_wav)
mel_spec_woman = wav2mel(woman_wav)
sex_m = 'man'
sex_w = 'woman'
mel2stft(mel_spec_man, sex_m, save_dir)
mel2stft(mel_spec_woman, sex_w, save_dir)


# y, sr = librosa.load(librosa.util.example_audio_file(), duration=5, offset=10)
# S = np.abs(librosa.stft(y))
# mel_spec = librosa.feature.melspectrogram(S=S, sr=sr)
# S_inv = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr)
#
# print("S shape is {} mel_spec shape is {} stft shape is {}".format(np.shape(S), np.shape(mel_spec), np.shape(stft)))
# audio = librosa.core.griffinlim(stft)
# print("audio shape is :",np.shape(audio))
# sf.write('./test.wav', audio, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
'''