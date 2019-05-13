import librosa
import numpy as np
import scipy.fftpack
import soundfile as sf
import extract_Mel_Spectrogram
import os
import matplotlib.pyplot as plt




def plotting_sepc(origin, stft, inverse):
    y1, fs = librosa.load(origin)
    time1 = np.linspace(0, len(y1)/fs, len(y1))
    # plt.figure()
    plt.figure(figsize=(10, 4))
    plt.subplot(3, 1, 1)
    plt.plot(time1, y1)

    # librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max, top_db=None), y_axis='log', x_axis='time')
    # plt.colorbar()
    plt.title('Original')

    y2, fs = librosa.load(stft)
    time2 = np.linspace(0, len(y2) / fs, len(y2))

    plt.subplot(3, 1, 2)
    plt.plot(time2, y2)
    # librosa.display.specshow(librosa.amplitude_to_db(np.abs(S_inv - S), ref=S.max(), top_db=None), vmax=0, y_axis='log',
    #                          x_axis='time', cmap='magma')
    plt.title('Stft')
    # plt.colorbar()
    y3, fs = librosa.load(inverse)
    time3 = np.linspace(0, len(y3) / fs, len(y3))
    plt.subplot(3, 1, 3)
    plt.plot(time3, y3)
    plt.title('Inverse')
    plt.tight_layout()
    plt.savefig('compare original, stft, inverse.png')

# man_original_data = '/mnt/junewoo/workspace/transform/test_wav/man_wav1.wav'
data1 = '/mnt/junewoo/workspace/transform/test_wav/man_8k_nfft0.016_inverse.wav'
# man_stft = '/mnt/junewoo/workspace/transform/test_wav/manS_stft_istft.wav'
# man_inverse ='/mnt/junewoo/workspace/transform/test_wav/man1.wav'
data2 = '/mnt/junewoo/workspace/transform/test_wav/man_8k_nfft0.032_inverse.wav'
data3 = '/mnt/junewoo/workspace/transform/test_wav/man_8k_nfft0.064_inverse.wav'
# woman_original_data = '/mnt/junewoo/workspace/transform/test_wav/woman_wav1.wav'
# woman_stft = '/mnt/junewoo/workspace/transform/test_wav/womanS_stft_istft.wav'
# woman_inverse = '/mnt/junewoo/workspace/transform/test_wav/woman1.wav'

# plotting_sepc(data1, data2, woman_inverse)
# plotting_sepc(woman_original_data, woman_stft, woman_inverse)