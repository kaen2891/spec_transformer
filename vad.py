# -*- coding: utf-8 -*-
from pyvad import vad,trim
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os
#from scipy.io import wavfile
'''
# for picture extract part
wav = '/data/dataset/IEMOCAP_wavonly/IEMOCAP_Wavonly/Wav/Ses01F_impro01/Ses01F_impro01_F005.wav'
(_, file_id) = os.path.split(wav)
print(file_id)

data, fs = librosa.load(wav)

time = np.linspace(0, len(data)/fs, len(data)) # time axis
plt.plot(time, data)
plt.title('Ses01F_impro01_F005.wav')
plt.savefig('input.png')
plt.show()

vact = vad(data, fs, fs_vad = 16000, hoplength = 30, vad_mode=3)

fig, ax1 = plt.subplots()

ax1.plot(time, data, color = 'b', label='speech waveform')
ax1.set_xlabel("TIME [s]")

ax2=ax1.twinx()
ax2.plot(time, vact, color="r", label = 'vad')
plt.yticks([0, 1] ,('unvoice', 'voice'))
ax2.set_ylim([-0.01, 1.01])

plt.legend()
plt.title('VAD Ses01F_impro01_F005.wav')
plt.savefig('vad.png')
#plt.title('VAD ')
plt.show()

trimed = trim(data, fs, fs_vad = 16000, hoplength = 30, vad_mode=3)

time = np.linspace(0, len(trimed)/fs, len(trimed)) # time axis
fig, ax1 = plt.subplots()

ax1.plot(time, trimed, color = 'b', label='speech waveform')
ax1.set_xlabel("TIME [s]")
plt.title('After trim Ses01F_impro01_F005.wav')
plt.savefig('after_cut_vad.png')
plt.show()
'''
'''
#if type(trimed) == NoneType:
if isinstance(trimed, type(None)):
    print("It can't VAD")
    data = librosa.core.resample(y, 22050, 16000)
else:
    print("VAD Finished...")
    data = librosa.core.resample(trimed, 22050, 16000)

out_put_dir = '/data/dataset/vad_result_mode3/'+emotion+'/'
#print(out_put_dir)
if not os.path.isdir(out_put_dir):
    os.mkdir(out_put_dir)
sf.write(out_put_dir+file_id, data, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
print("{} is saved".format(out_put_dir+file_id))

'''


def june_vad(save_dir,speaker_id,wav_dir):
    #name = '/data/Ses01F_script01_1_F012.wav'
    (_, file_id) = os.path.split(wav_dir)
    print(file_id)

    y, fs = librosa.load(wav_dir)

    trimed = trim(y, fs, fs_vad = 16000, hoplength = 30, vad_mode=3)

    if isinstance(trimed, type(None)):
        print("It can't VAD")
        data = librosa.core.resample(y, 22050, 16000)
    else:
        print("VAD Finished...")
        data = librosa.core.resample(trimed, 22050, 16000)

    #now_dir = os.getcwd()
    output_dir = os.path.join(save_dir,speaker_id)
    #print(out_put_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sf.write(output_dir+'/'+file_id, data, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
    print("{} is saved".format(output_dir+file_id))

'''

#start_time = time()

'''
#end_time = time()

#time_taken = end_time - start_time
#print(time_taken)

