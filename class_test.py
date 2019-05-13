import mel_to_inverse
import numpy as np
import librosa
import librosa.display
import os
import extract_Mel_Spectrogram
import matplotlib.pyplot as plt
# (self, wav, mel_spec, frame_length, frame_stride, input_nfft, input_stride, sr, save_dir, gender):
# test_wav = '/mnt/junewoo/workspace/transform/test_wav/man_wav1.wav'
test_wav = '/mnt/junewoo/workspace/transform/16k.wav'
sr = 16000
input_nfft = 0.032
input_stride = 0.010
hop_length = int(round(sr*input_stride))
save_dir ='/mnt/junewoo/workspace/transform/test_wav/'
# mel_spec = extract_Mel_Spectrogram.Mel_S(test_wav, sr, input_nfft, input_stride)

file_save_name = 'man_16k_nfft0.032'
mel = extract_Mel_Spectrogram.Mel_S(test_wav, sr, input_nfft, input_stride)
extract_Mel_Spectrogram.Mel_figure(mel, sr, input_nfft, input_stride, " ")
print("mel shape is in test:", np.shape(mel))
print("mel type is in test:", type(mel))
height = len(mel) # y axis: n_mels dim
width = len(mel.T) # x axis: tiem step
print("height {} width {}".format(height, width))

# get scalar of mel-spec
test_dim = mel[:, 0]
for i in range(len(mel.T)):
    test_dim = mel[:, i]
    print("{} th check".format(i))
# print(len(mel[:, 0]))
print("test")

# get log mel scale
log_m = librosa.amplitude_to_db(mel)
extract_Mel_Spectrogram.Mel_figure(log_m, sr, input_nfft, input_stride, "log_mel_scale")
# back to db -> amplitude
# librosa
for k in range(len(log_m.T)):
    log_dim = log_m[:, k]
    print("{} th check".format(k))
test = mel_to_inverse.Convert2Wav(wav=test_wav, mel_spec=mel, frame_length=input_nfft, frame_stride=input_stride, sr=sr, save_dir=save_dir, save_name=file_save_name)
mel2 = test.mel2wav()

# log mel -> mel_spec convert
back_to_amp = librosa.db_to_amplitude(log_m)
for t in range(len(back_to_amp.T)):
    back_dim = back_to_amp[:, t]
    print("{} th check".format(t))
extract_Mel_Spectrogram.Mel_figure(back_to_amp, sr, input_nfft, input_stride, "log_to_origin")
# test
 # plot figure
