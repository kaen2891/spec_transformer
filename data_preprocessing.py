import vad
import extract_Mel_Spectrogram
import os
from glob import glob
import pickle
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import mel_to_inverse

frame_length = 0.032
frame_stride = 0.010
sr = 16000
length_of_nmel = 40

def save_vad_wav(glob_list, save_dir):
    for i in range(len(glob_list)):
        (dir, file_id) = os.path.split(glob_list[i])  # dir + filename.pcm
        (dir_dir, speaker_id) = os.path.split(dir)
        print("file_id is {} and speaker_id is {}".format(file_id, speaker_id))
        vad.june_vad(save_dir, speaker_id, glob_list[i])

def extract_phase_mag(data, fs):
    y, sr = librosa.load(data, sr=fs)
    D1 = librosa.stft(y)
    print("D1의 shape:", np.shape(D1))



def check_max_length(glob_list):
    ''' check wav max length  '''
    length_max = 0
    for i in range(len(glob_list)):
        y, sr = librosa.load(glob_list[i], sr=16000)
        wav_length = len(y) / sr
        if wav_length > length_max:
            length_max = wav_length
    return length_max

def extract_wav_Mel_Spectrogram(glob_list, fs, frame_length, frmae_stride):
    Mel_Spectrogram_padding_list = []
    ''' Extract Mel Spectrogram'''
    ''' It needs wav_file, fs, frame_length, frame_stride'''
    ''' Here, wav_file: glob_list[k], fs = 16000, frame_length = 0.032, frame_stride = 0.010'''
    for k in range(len(glob_list)):
        Mel_Spectrogram = extract_Mel_Spectrogram.Mel_S(glob_list[k], fs, frame_length, frmae_stride)
        Mel_Spectrogram_padding_list.append(Mel_Spectrogram)
    return Mel_Spectrogram_padding_list



def divide_tvt(glob_list, gender):
    train_filename = 'train_Mels'
    valid_filename = 'valid_Mels'
    test_filename = 'test_Mels'

    train_output = '/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/vad_result/train/'
    train_output_dir = os.path.join(train_output, gender)
    print("train output dir is ", train_output_dir)
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)

    valid_output = '/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/vad_result/valid/'
    valid_output_dir = os.path.join(valid_output, gender)
    if not os.path.exists(valid_output_dir):
        os.makedirs(valid_output_dir)

    test_output = '/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/vad_result/test/'
    test_output_dir = os.path.join(test_output, gender)
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    train_len = round(len(glob_list) * 0.7)
    valid_len = round(len(glob_list) * 0.15)
    test_len = len(glob_list) - (train_len + valid_len)

    train = []
    valid = []
    test = []
    for i in range(train_len):
        train.append(glob_list[i])
    for ii in range(valid_len):
        count = train_len + ii
        valid.append(glob_list[count])
    for iii in range(test_len):
        count = train_len + valid_len + iii
        test.append(glob_list[count])
    with open(train_output_dir+'/'+train_filename, 'wb') as fp:
        pickle.dump(train, fp)

    with open(valid_output_dir+'/'+valid_filename, 'wb') as fp:
        pickle.dump(valid, fp)

    with open(test_output_dir+'/'+test_filename, 'wb') as fp:
        pickle.dump(test, fp)

    # return train_label, valid_label, test_label
#
# def restore_MelS(input_Mels, name):
#     print("input Mels shape is :", np.shape(input_Mels))
#     y_hat = librosa.istft(input_Mels)
#
#     sf.write('./'+name, y_hat, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
#     # return y_hat

def down_sample(input_wav, origin_sr, resample_sr):
    y, sr = librosa.load(input_wav, sr=origin_sr)
    resample = librosa.resample(y, sr, resample_sr)
    print("original wav sr: {}, original wav shape: {}, resample wav sr: {}, resmaple shape: {}".format(origin_sr, y.shape, resample_sr, resample.shape))

    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    time1 = np.linspace(0, len(y) / sr, len(y))
    plt.plot(time1, y)
    plt.title('Original Wav')

    plt.subplot(2, 1, 2)
    time2 = np.linspace(0, len(resample) / resample_sr, len(resample))
    plt.plot(time2, resample)
    plt.title('Resampled Wav')

    plt.tight_layout()
    plt.savefig('compare_16k_vs_8k.png')

    sf.write('./' + '16k.wav' , y, origin_sr, format='WAV', endian='LITTLE', subtype='PCM_16')
    sf.write('./' + '8k.wav', resample, resample_sr, format='WAV', endian='LITTLE', subtype='PCM_16')

def set_length_mel(mel_spec1, mel_spec2, len_n_mel):
    mel1 = len(mel_spec1[0])
    mel2 = len(mel_spec2[0])
    div_mel1 = int(mel1/100)
    div_mel2 = int(mel2/100)

    if div_mel1 != div_mel2:
        if div_mel1 > div_mel2:
            div_mel1 = div_mel2
        else:
            div_mel2 = div_mel1

    recover_legnth_a = div_mel1 * 100
    recover_legnth_b = div_mel2 * 100
    new_a = []
    new_b = []
    for i in range(len_n_mel):
        new_a.append(mel_spec1[i][:recover_legnth_a])
        new_b.append(mel_spec2[i][:recover_legnth_b])
    final_mel_spec_a = np.array(new_a)
    final_mel_spec_b = np.array(new_b)
    return final_mel_spec_a, final_mel_spec_b

def cut_and_save_mels(female_dataset, male_dataset, length_of_nmel, save_cut_dir, set_name):
    woman_set = []
    man_set = []
    for i in range(len(female_dataset)):
        female_mel = female_dataset[i]
        male_mel = male_dataset[i]
        set_a, set_b = set_length_mel(female_mel, male_mel, length_of_nmel)
        print("i {} male mel {} female mel {} cut male {} cut female{}".format(i,np.shape(male_mel), np.shape(female_mel), np.shape(set_a), np.shape(set_b)))
        woman_set.append(set_a)
        man_set.append(set_b)
    if not os.path.exists(save_cut_dir):
        os.makedirs(save_cut_dir)
    with open(save_cut_dir+'/'+set_name+'female', 'wb') as fp:
        pickle.dump(woman_set, fp)
    with open(save_cut_dir+'/'+set_name+'male', 'wb') as fp:
        pickle.dump(man_set, fp)

    for k in range(len(woman_set)):
        file_save_name = 'cut_female{}'.format(k+1)
        woman_mel_spec = mel_to_inverse.Convert2Wav(wav=None, mel_spec=woman_set[k], frame_length=frame_length,
                                                  frame_stride=frame_stride, sr=sr, save_dir=save_cut_dir,
                                                  save_name=file_save_name)
        woman_mel_spec.mel2stft()

        file_save_name = 'cut_male{}'.format(k + 1)
        man_mel_spec = mel_to_inverse.Convert2Wav(wav=None, mel_spec=man_set[k], frame_length=frame_length,
                                                  frame_stride=frame_stride, sr=sr, save_dir=save_cut_dir,
                                                  save_name=file_save_name)
        man_mel_spec.mel2stft()


def data_preprop():

    # Down-Sampling Test
    # man_original_data = '/mnt/junewoo/workspace/transform/test_wav/man_wav1.wav'
    # down_sample(man_original_data, 16000, 8000)


    # load all wav files and process with vad (cut silence)
    public_data = glob('/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/public/*/*.wav') #from data folder. change dir
    vad_save_dir = '/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/public_vad_result/' #from data folder. change dir
    save_vad_wav(public_data, vad_save_dir)

    # check max legnth of all wavs
    vad_wav = glob('/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/public_vad_result/*/*.wav')
    check_length = check_max_length(vad_wav)

    #extract for Mel_Spectrogram
    Mel_S = extract_wav_Mel_Spectrogram(vad_wav, sr, frame_length, frame_stride)

    # save all mel_spec
    # with open('./Mel_S', 'wb') as fp:
    #     pickle.dump(Mel_S, fp)

    # load all mel_spec
    # with open('./Mel_S', 'rb') as file:
    #     Mel_Spec = pickle.load(file)

    # get 1 pair(female, male) dataset
    vad_female_data = sorted(glob('/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/public_vad_result/falys0/*.wav'))
    vad_male_data = sorted(glob('/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/public_vad_result/masch0/*.wav'))

    # get mel_spectrogram
    female_Mel_S = extract_wav_Mel_Spectrogram(vad_female_data, sr, frame_length, frame_stride)
    male_Mel_S = extract_wav_Mel_Spectrogram(vad_male_data, sr, frame_length, frame_stride)

    # divide train / valid / test set (0.7 : 0.15 : 0.15)
    gender = 'female'
    divide_tvt(female_Mel_S, gender)
    gender = 'male'
    divide_tvt(male_Mel_S, gender)

    # save female, male train/valid/test set
    with open('/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/vad_result/train/male/train_Mels', 'rb') as file:
        train_Male = pickle.load(file)
    with open('/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/vad_result/train/female/train_Mels', 'rb') as file:
        train_Female = pickle.load(file)
    with open('/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/vad_result/valid/male/valid_Mels', 'rb') as file:
        valid_Male = pickle.load(file)
    with open('/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/vad_result/valid/female/valid_Mels', 'rb') as file:
        valid_Female = pickle.load(file)
    with open('/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/vad_result/test/male/test_Mels', 'rb') as file:
        test_Male = pickle.load(file)
    with open('/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/vad_result/test/female/test_Mels', 'rb') as file:
        test_Female = pickle.load(file)

    # equal 2 dataset
    cut_dir = '/mnt/junewoo/workspace/transform/pair_1_dataset/'
    set_name = 'train'
    save_dir = os.path.join(cut_dir, set_name)
    cut_and_save_mels(train_Female, train_Male, length_of_nmel, save_dir, set_name)
    print("train finish")
    set_name = 'valid'
    save_dir = os.path.join(cut_dir, set_name)
    cut_and_save_mels(valid_Female, valid_Male, length_of_nmel, save_dir, set_name)
    print("valid finish")
    set_name = 'test'
    save_dir = os.path.join(cut_dir, set_name)
    cut_and_save_mels(test_Female, test_Male, length_of_nmel, save_dir, set_name)
    print("test finish")


    #
    # # origin_a
    # save_dir = '/mnt/junewoo/workspace/transform/test_wav/cut_shape/'
    # file_save_name = 'ori_mel_female'
    # a_ori_test = mel_to_inverse.Convert2Wav(wav=None, mel_spec=train_Female[0], frame_length=input_nfft, frame_stride=input_stride, sr=sr, save_dir=save_dir, save_name=file_save_name)
    # a_ori_test.mel2stft()
    # # cut_a
    # # print("check:", new_Mel_train_a.shape[0])
    # save_dir = '/mnt/junewoo/workspace/transform/test_wav/cut_shape/'
    # file_save_name = 'cut_mel_female'
    # a_test = mel_to_inverse.Convert2Wav(wav=None, mel_spec=cut_a, frame_length=input_nfft, frame_stride=input_stride, sr=sr, save_dir=save_dir, save_name=file_save_name)
    # a_test.mel2stft()
    #
    # # origin_b
    # save_dir = '/mnt/junewoo/workspace/transform/test_wav/cut_shape/'
    # file_save_name = 'ori_mel_male'
    # b_ori_test = mel_to_inverse.Convert2Wav(wav=None, mel_spec=train_Male[0], frame_length=input_nfft, frame_stride=input_stride, sr=sr, save_dir=save_dir, save_name=file_save_name)
    # b_ori_test.mel2stft()
    # # cut_b
    # save_dir = '/mnt/junewoo/workspace/transform/test_wav/cut_shape/'
    # file_save_name = 'cut_mel_male'
    # b_test = mel_to_inverse.Convert2Wav(wav=None, mel_spec=cut_b, frame_length=input_nfft, frame_stride=input_stride, sr=sr, save_dir=save_dir, save_name=file_save_name)
    # b_test.mel2stft()

if __name__ == '__main__':
    data_preprop()
