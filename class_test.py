import mel_to_inverse
import numpy as np
import librosa
import os
import extract_Mel_Spectrogram
# (self, wav, mel_spec, frame_length, frame_stride, input_nfft, input_stride, sr, save_dir, gender):
# test_wav = '/mnt/junewoo/workspace/transform/test_wav/man_wav1.wav'
test_wav = '/mnt/junewoo/workspace/transform/16k.wav'
sr = 16000
input_nfft = 0.032
input_stride = 0.010
save_dir ='/mnt/junewoo/workspace/transform/test_wav/'
mel_spec = extract_Mel_Spectrogram.Mel_S(test_wav, sr)

file_save_name = 'man_16k_nfft0.032_nmel80'
test = mel_to_inverse.Convert2Wav(wav=test_wav, mel_spec=mel_spec, frame_length=input_nfft, frame_stride=input_stride, sr=sr, save_dir=save_dir, save_name=file_save_name)
mel = test.wav2mel()

print("mel shape is in test:", np.shape(mel))
print("mel type is in test:", type(mel))
height = len(mel)
width = len(mel.T)
print("height {} width {}".format(height, width))

env = mel[0] # 336(time step)
print(env)
print(len(env))
print((mel[0][0]))
print("height length is {} and width length is {}".format(height, width))
test_dim = mel[:, 0]
for i in range(len(mel.T)):
    test_dim = mel[:, i]
    print("{} th check".format(i))
# print(len(mel[:, 0]))
print("test")
# mel_to_inverse.Convert2Wav()
# print("mel shape is :", np.shape(mel))
# log_S = librosa.logamplitude(S, ref_power=np.max)
# log_m = librosa.amplitude_to_db(mel, ref=np.max)
log_m = librosa.amplitude_to_db(mel)
print(np.shape(log_m))
# back to db -> amplitude
# librosa
for k in range(len(log_m.T)):
    log_dim = log_m[:, k]
    print("{} th check".format(k))
# mel2 = test.spec2wav()
# print("finish spec2wav")
back_to_amp = librosa.db_to_amplitude(log_m)
for t in range(len(back_to_amp.T)):
    back_dim = back_to_amp[:, t]
    print("{} th check".format(t))
#
# mel3 = test.mel2stft()
# print("finish spec3wav")

