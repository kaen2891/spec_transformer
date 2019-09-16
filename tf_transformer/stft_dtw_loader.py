import pickle
import numpy as np
import librosa
# from ezdtw import dtw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import librosa.display
from pyvad import vad,trim
import os
from glob import glob

from scipy.spatial.distance import cdist

nfft=512
hop=256

def dtw_distance(distances):
    DTW = np.empty_like(distances)
    # DTW[:, 0] = np.inf
    # DTW[0, :] = np.inf
    DTW[0, 0] = 0
    print(DTW.shape[0], DTW.shape[1]) # 10, 8
    for i in range(0, DTW.shape[0]):
        for j in range(0, DTW.shape[1]):
            if i ==0 and j==0:
                DTW[i, j] = distances[i, j]
            elif i == 0:
                DTW[i, j] = distances[i, j] + DTW[i, j-1]
            elif j == 0:
                DTW[i, j] = distances[i, j] + DTW[i-1, j]
            else:
                DTW[i, j] = distances[i, j] + min(DTW[i-1, j],  # insertion
                                              DTW[i, j-1],  # deletion
                                              DTW[i-1, j-1] #match
                                             )
    #print("done")
    return DTW

def backtrack(DTW):
    i, j = DTW.shape[0] - 1, DTW.shape[1] - 1
    output_x = []
    output_y = []
    output_x.append(i) # last scalar in reference spectrogram
    output_y.append(j) # last scalar in compare spectrogram
    while i > 0 and j > 0:
        local_min = np.argmin((DTW[i - 1, j - 1], DTW[i, j - 1], DTW[i - 1, j]))
        if local_min == 0:
            i -= 1
            j -= 1
        elif local_min == 1:
            j -= 1
        else:
            i -= 1
        output_x.append(i)
        output_y.append(j)

    output_x.append(0)
    output_y.append(0)
    output_x.reverse()
    output_y.reverse()
    return np.array(output_x), np.array(output_y) # this output is dtw path. Longer than reference

def multi_DTW(a,b,len_ref,len_tar): #a is ref, b is compare. a, b lengths are same.
    cnt = []
    for x in range(len(a)):
        if a[x-1] == a[x]:
            cnt.append(x)
    target = np.delete(b, cnt) # 470, len_ref = 473
    if len(target) < len_ref:
        differ =  len_ref - len(target) # 3
        target = np.pad(target,(0,differ), 'constant', constant_values=(1, len_tar-1))
    return target
    # print("done")
    # print("done")

def my_dtw(a, b, len_ref, len_tar, distance_metric='euclidean'):
    distance = cdist(a, b, distance_metric)
    cum_min_dist = dtw_distance(distance)
    x, y = backtrack(cum_min_dist)
    final_y = multi_DTW(x,y,len_ref,len_tar)
    #print(final_y, len(final_y))
    # print("here, done")
    return final_y

def cut_small_value(magnitude):
    mask = (magnitude >= 1e-2).astype(np.float32)
    new_mag = magnitude * mask
    new_mag[new_mag <= 1e-2] = 1e-3

    return new_mag

def plot_wav(spectrogram, speaker_name, mode, save_dir):
    #input_istft = librosa.istft(spectrogram, hop_length=hop)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), y_axis='hz', x_axis='time', sr=16000, hop_length=hop)
    name = speaker_name
    plt.title(name)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_dir + name + '.png')

def make_wav(spectrogram, speaker_name, mode, save_dir):
    name = speaker_name
    input_istft = librosa.istft(spectrogram, hop_length=hop)
    sf.write(save_dir + name + '.wav', input_istft, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')

def check_array(array):
    print("hi")

#from librosa.effects import trim, split
def use_trim(y, frame, hop):
    test = librosa.effects.trim(y, frame_length=512, hop_length=256)
    return test

def return_mag_pha(input_stft):
    mag, pha = librosa.magphase(input_stft)
    return mag, pha

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

def zero_padding(array, max_length):
    differ = len(max_length.T) - len(array.T)
    npad = ((0, 0), (0, differ))
    target = np.pad(array, npad, 'constant')

    return target
def zero_padding_complex(complex_array, max_length):
    differ = len(max_length.T) - len(complex_array.T)
    complex_npad = ((0,0),(0,differ))
    complex_target = np.pad(complex_array, complex_npad, 'constant')
    return complex_target

def save_vad_wav(glob_list, save_dir):
    for i in range(len(glob_list)):
        (dir, file_num) = os.path.split(glob_list[i])  # dir + filename
        (dir_dir, speaker_id) = os.path.split(dir)
        (_, speaker_name) = os.path.split(dir_dir)
        print("file_num is {} and speaker_name is {}".format(file_num, speaker_name))
        file_num = file_num[7:12]
        speaker_name = speaker_name[7:10]
        #print(file_num, speaker_name)
        cmu_vad(save_dir, file_num, speaker_name, glob_list[i])

def cmu_vad(save_dir,file_num, speaker_id,wav_dir):
    #name = '/data/Ses01F_script01_1_F012.wav'
    (_, file_id) = os.path.split(wav_dir)
    print(file_id)

    y, fs = librosa.load(wav_dir, sr=16000)

    trimed = trim(y, fs, fs_vad = fs, hoplength = 30, vad_mode=3)

    if isinstance(trimed, type(None)):
        print("It can't VAD")
        data = y
    else:
        print("VAD Finished...")
        data = trimed

    #now_dir = os.getcwd()
    name = speaker_id+'_'+file_num+'.wav'


    #print(out_put_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sf.write(save_dir+name, data, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
    print("{} is saved".format(save_dir+'/'+name))

def last_process():
    # X_glob = sorted(glob('./dataset/*_male_utt_hop{}/phase'.format(hop))) # 1e-3, 0 padding
    X_glob = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/0911/*_male_utt_hop{}/cut_small_value_1'.format(hop)))
    Y_glob = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/0911/*_female_utt_hop{}/cut_small_value_1'.format(hop)))
    X_glob2 = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/0911/*_male_utt_hop{}/cut_small_value_2'.format(hop)))
    Y_glob2 = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/0911/*_female_utt_hop{}/cut_small_value_2'.format(hop)))
    X_phase = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/0911/*_male_utt_hop{}/phase'.format(hop)))
    Y_phase = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/0911/*_female_utt_hop{}/phase'.format(hop)))
    # X_glob = sorted(glob('./dataset/*_male_utt_hop{}/cut_small_value_2'.format(hop))) # all 0 padding
    # X_glob = sorted(glob('./dataset/*_male_utt_hop{}/original_mag'.format(hop)))
    # Y_glob = sorted(glob('./dataset/*_female_utt_hop{}/original_mag'.format(hop)))
    # Y_glob = sorted(glob('./dataset/*_female_utt_hop{}/phase'.format(hop)))# 1e-3, 0 padding
    # Y_glob = sorted(glob('./dataset/*_female_utt_hop{}/cut_small_value_2'.format(hop)))# 1all 0 padding
    # data_num = len(X_glob) + 1
    '''
    with open(X_glob[0], 'rb') as file:
        for_shape = pickle.load(file)
    '''
    with open(X_phase[0], 'rb') as file2:
        for_phase = pickle.load(file2)

    '''
    X_train = np.zeros_like(for_shape)
    Y_train = np.zeros_like(for_shape)
    X_test = np.zeros_like(for_shape)
    Y_test = np.zeros_like(for_shape)
    X_train2 = np.zeros_like(for_shape)
    Y_train2 = np.zeros_like(for_shape)
    X_test2 = np.zeros_like(for_shape)
    Y_test2 = np.zeros_like(for_shape)
    '''
    X_train_phase = np.zeros_like(for_phase)
    Y_train_phase = np.zeros_like(for_phase)
    X_test_phase = np.zeros_like(for_phase)
    Y_test_phase = np.zeros_like(for_phase)
    '''
    for i in range(len(X_glob)):
        # with open('/mnt/junewoo/workspace/transform/test_head/x_train', 'rb') as file:
        #     X_train = pickle.load(file)
        with open(X_glob[i], 'rb') as file:
            read_male = pickle.load(file)
        with open(Y_glob[i], 'rb') as file2:
            read_female = pickle.load(file2)
        with open(X_glob2[i], 'rb') as file:
            read_male2 = pickle.load(file)
        with open(Y_glob2[i], 'rb') as file2:
            read_female2 = pickle.load(file2)
        divide_rate = 900
        if i < divide_rate:
            X_train = np.append(X_train,read_male, axis=0)
            Y_train = np.append(Y_train,read_female, axis=0)
            X_train2 = np.append(X_train2, read_male2, axis=0)
            Y_train2 = np.append(Y_train2, read_female2, axis=0)
        else:
            X_test = np.append(X_test, read_male, axis=0)
            Y_test = np.append(Y_test, read_female, axis=0)
            X_test2 = np.append(X_test2, read_male2, axis=0)
            Y_test2 = np.append(Y_test2, read_female2, axis=0)
        print("{} finished".format(i))
    '''

    for k in range(len(X_phase)):
        # with open('/mnt/junewoo/workspace/transform/test_head/x_train', 'rb') as file:
        #     X_train = pickle.load(file)
        with open(X_phase[k], 'rb') as file:
            read_male = pickle.load(file)
        with open(Y_phase[k], 'rb') as file2:
            read_female = pickle.load(file2)
        divide_rate = 900
        if k < divide_rate:
            X_train_phase = np.append(X_train_phase, read_male, axis=0)
            Y_train_phase = np.append(Y_train_phase, read_female, axis=0)
        else:
            X_test_phase = np.append(X_test_phase, read_male, axis=0)
            Y_test_phase = np.append(Y_test_phase, read_female, axis=0)
        print("{} finished".format(k))

    '''
    X_train = X_train[1:,:,:]
    Y_train = Y_train[1:,:,:]
    X_test = X_test[1:,:,:]
    Y_test = Y_test[1:,:,:]

    X_train2 = X_train2[1:,:,:]
    Y_train2 = Y_train2[1:,:,:]
    X_test2 = X_test2[1:,:,:]
    Y_test2 = Y_test2[1:,:,:]
    '''
    X_train_phase = X_train_phase[1:, :, :]
    Y_train_phase = Y_train_phase[1:, :, :]
    X_test_phase = X_test_phase[1:, :, :]
    Y_test_phase = Y_test_phase[1:, :, :]
    print('done')
    # x_train_dir = '/mnt/junewoo/speech/cmu_all_dataset/0911/x_train_cut_small_value_1_{}'.format(hop)
    x_train_dir = './dataset/x_train_phase_{}'.format(hop)
    # y_train_dir = '/mnt/junewoo/speech/cmu_all_dataset/0911/y_train_cut_small_value_1_{}'.format(hop)
    y_train_dir = './dataset/y_train_phase_{}'.format(hop)
    # x_test_dir = '/mnt/junewoo/speech/cmu_all_dataset/0911/x_test_cut_small_value_1_{}'.format(hop)
    x_test_dir = './dataset/x_test_phase_{}'.format(hop)
    # y_test_dir = '/mnt/junewoo/speech/cmu_all_dataset/0911/y_test_cut_small_value_1_{}'.format(hop)
    y_test_dir = './dataset/y_test_phase_{}'.format(hop)
    '''
    x_train_dir2 = '/mnt/junewoo/speech/cmu_all_dataset/0911/x_train_cut_small_value_2_{}'.format(hop)
    y_train_dir2 = '/mnt/junewoo/speech/cmu_all_dataset/0911/y_train_cut_small_value_2_{}'.format(hop)
    x_test_dir2 = '/mnt/junewoo/speech/cmu_all_dataset/0911/x_test_cut_small_value_2_{}'.format(hop)
    y_test_dir2 = '/mnt/junewoo/speech/cmu_all_dataset/0911/y_test_cut_small_value_2_{}'.format(hop)
    '''
    # y_test_dir = './dataset/y_test_phase_{}'.format(hop)
    # x_pha_dir = '/mnt/junewoo/speech/cmu_all_dataset/0911/x_train_phase_{}'.format(hop)
    # y_pha_dir = '/mnt/junewoo/speech/cmu_all_dataset/0911/y_train_phase_{}'.format(hop)
    with open(x_train_dir, 'wb') as fp:
        pickle.dump(X_train_phase, fp)
    with open(y_train_dir, 'wb') as fp:
        pickle.dump(Y_train_phase, fp)
    with open(x_test_dir, 'wb') as fp:
        pickle.dump(X_test_phase, fp)
    with open(y_test_dir, 'wb') as fp:
        pickle.dump(Y_test_phase, fp)

    # with open(x_pha_dir, 'wb') as fp:
    #    pickle.dump(X_phase_all, fp)
    # with open(y_pha_dir, 'wb') as fp:
    #    pickle.dump(Y_phase_all, fp)
    '''
    with open(x_train_dir2, 'wb') as fp:
        pickle.dump(X_train2, fp)
    with open(y_train_dir2, 'wb') as fp:
        pickle.dump(Y_train2, fp)
    with open(x_test_dir2, 'wb') as fp:
        pickle.dump(X_test2, fp)
    with open(y_test_dir2, 'wb') as fp:
        pickle.dump(Y_test2, fp)
    '''




max_all = []
min_all = []

all_data = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/vad/aew/*.wav'))
# find maximum wav
for b in range(len(all_data)):
    b = b + 1
    if b <= 593:
        char = 'a'
        if b < 10:
            k = '000' + str(b)
        elif b >= 10 and b < 100:
            k = '00' + str(b)
        elif b >= 100:
            k = '0' + str(b)
    elif b > 593:
        char = 'b'
        b = b-593
        if b < 10:
            k = '000' + str(b)
        elif b >= 10 and b < 100:
            k = '00' + str(b)
        elif b >= 100:
            k = '0' + str(b)
    read_file = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/vad/*/*{}{}.wav'.format(char,k)))
    length = []
    for x in range(len(read_file)):
        # print(read_file[x])
        y, sr = librosa.load(read_file[x], sr=16000)
        du = librosa.core.get_duration(y=y, sr=sr, n_fft=nfft, hop_length=hop)
        length.append(du)
    np_length = np.asarray(length) # convert list -> nd array
    max_len = np_length.max()
    min_len = np_length.min()
    # avg_len = np_length.mean() # find average
    avg_idx, idx = find_nearest(np_length, max_len) # find nearest stft from average
    avg_idx2, idx2 = find_nearest(np_length, min_len)  # find nearest stft from average
    max_utt = read_file[idx]
    min_utt = read_file[idx2]
    max_all.append(max_utt)
    min_all.append(min_utt)
length = []

'''find min spectrogram'''
for c in range(len(min_all)):
    x, sr = librosa.load(min_all[c], sr=16000)
    du = librosa.core.get_duration(y=x, sr=sr, n_fft=nfft, hop_length=hop)
    length.append(du)

'''test'''

np_length = np.asarray(length) # convert list -> nd array
min_len = np_length.min() # find average
avg_idx, idx = find_nearest(np_length, min_len) # find nearest stft from average
all_min_utt = min_all[idx]
#all_max_utt = max_all[idx]
print("whole maximum file is {} and the second is {}".format(all_min_utt, min_len))
z, sr = librosa.load(all_min_utt, sr=16000)
z_len = librosa.core.get_duration(y=z, sr=16000, n_fft=nfft, hop_length=hop)
maxlen_stft = librosa.stft(z, n_fft=nfft, hop_length=hop)
maxlen_mag, maxlen_pha = return_mag_pha(maxlen_stft)
print(np.shape(maxlen_stft))


'''find max spectrogram'''
for c in range(len(max_all)):
    x, sr = librosa.load(max_all[c], sr=16000)
    du = librosa.core.get_duration(y=x, sr=sr, n_fft=nfft, hop_length=hop)
    length.append(du)
np_length = np.asarray(length) # convert list -> nd array
max_len = np_length.max() # find average
avg_idx, idx = find_nearest(np_length, max_len) # find nearest stft from average
all_max_utt = max_all[idx]
print("whole maximum file is {} and the second is {}".format(all_max_utt, max_len))
z, sr = librosa.load(all_max_utt, sr=16000)
z_len = librosa.core.get_duration(y=z, sr=16000, n_fft=nfft, hop_length=hop)
maxlen_stft = librosa.stft(z, n_fft=nfft, hop_length=hop)
maxlen_mag, maxlen_pha = return_mag_pha(maxlen_stft)
print(np.shape(maxlen_stft))

print('done')
print('-----')
a = 0
for i in range(len(all_data)):
    i = i + 1
    a = i
    if a < 10:
        a = '000' + str(a)
    elif a >= 10 and a < 100:
        a = '00' + str(a)
    elif a >= 100 and a < 1000:
        a = '0' + str(a)
    elif a > 1000:
        a = str(a)
    if i <= 593:
        char = 'a'
        if i < 10:
            t = '000' + str(i)
        elif i >= 10 and i < 100:
            t = '00' + str(i)
        elif i >= 100:
            t = '0' + str(i)
    elif i > 593:
        char = 'b'
        #a = i
        i = i - 593
        if i < 10:
            t = '000' + str(i)
        elif i >= 10 and i < 100:
            t = '00' + str(i)
        elif i >= 100:
            t = '0' + str(i)

    read_file = sorted(glob('/mnt/junewoo/speech/cmu_all_dataset/vad/*/*{}{}.wav'.format(char, t)))
    length = []

    for x in range(len(read_file)):
        # print(read_file[x])
        y, sr = librosa.load(read_file[x], sr=16000)
        du = librosa.core.get_duration(y=y, sr=sr, n_fft=nfft, hop_length=hop)
        length.append(du)
    if length[0] > length[1]:
        reference_utt = read_file[0]
    else:
        reference_utt = read_file[1]
        print("length[0] is more big")
    print("more big one is ", reference_utt)
    ###
    #dtw with many samples....
    #np_length = np.asarray(length) # convert list -> nd array
    #avg_len = np_length.mean() # find average
    #avg_idx, idx = find_nearest(np_length, avg_len) # find nearest stft from average
    ###
    

    #reference_utt = read_file[idx]
    stand, sr = librosa.load(reference_utt, sr=16000)  # reference => average
    refer_stft = librosa.stft(stand, n_fft=nfft, hop_length=hop)  # reference stft

    male_infor_array = []
    male_final_array = []
    male_phase_array = []
    female_infor_array = []
    female_final_array = []
    female_phase_array = []

    #os 부분 생성
    dir = '/mnt/junewoo/speech/cmu_all_dataset/0911/'

    male_data_name = str(a)+'_male_utt_hop'+str(hop)
    female_data_name = str(a)+'_female_utt_hop'+str(hop)

    female_dir = os.path.join(dir, female_data_name)
    if not os.path.isdir(female_dir):
        os.makedirs(female_dir)
    male_dir = os.path.join(dir, male_data_name)
    if not os.path.isdir(male_dir):
        os.makedirs(male_dir)

    for k in range(len(read_file)): # original mag
        if k != 0:  # female
            y, sr = librosa.load(read_file[k], sr=16000)  # DTW compare load
            (wav_check1, file_number) = os.path.split(read_file[k])
            (wav_check2, speaker_id) = os.path.split(wav_check1)
            file_number = file_number[:-4]
            compare_stft = librosa.stft(y, n_fft=nfft, hop_length=hop)  # DTW compare source
            output = my_dtw(refer_stft.T, compare_stft.T, len(refer_stft.T), len(compare_stft.T))  # using dtw, output is path
            print("k = {}, id = {}, female compare_stft len = {}, output len = {}".format(k,speaker_id,len(compare_stft.T), len(output)))
            final_output = compare_stft[:, output]  # compare source -> adapt path
            origin_mag, origin_pha = return_mag_pha(final_output)
            #pad_stft = zero_padding_complex(final_output, maxlen_stft)
            #mag_pad, pha_pad = return_mag_pha(pad_stft)
            cut_small_mag = cut_small_value(origin_mag)  # cut less than 1e-2 -> 1e-3
            # zero padding with origin mag, pha(complex type), small_value
            origin_mag_pad = zero_padding(origin_mag, maxlen_mag)
            origin_pha_pad = zero_padding_complex(origin_pha, maxlen_pha)
            small_mag_pad_1 = zero_padding(cut_small_mag, maxlen_mag)
            small_mag_pad_2 = cut_small_value(origin_mag_pad)
            #save original zero padding to plot, wav
            mode = 'origin_zero_padding_data'
            original_female_savedir = os.path.join(dir, female_data_name, mode, '')
            if not os.path.isdir(original_female_savedir):
                os.makedirs(original_female_savedir)
            plot_wav(final_output,file_number,'', original_female_savedir)
            make_wav(final_output,file_number,'', original_female_savedir)
            # save cut_small_value zero padding to plot, wav
            mode = 'cut_small_zero_padding_data_1' # 1e-3, 0 also have
            cut_small_female_savedir_1 = os.path.join(dir, female_data_name, mode, '')
            if not os.path.isdir(cut_small_female_savedir_1):
                os.makedirs(cut_small_female_savedir_1)
            plot_wav(small_mag_pad_1, file_number, '', cut_small_female_savedir_1)
            make_wav(small_mag_pad_1, file_number, '', cut_small_female_savedir_1)
            # female_infor_array.append(speaker_id)  # append speaker_id to list, 0

            mode = 'cut_small_zero_padding_data_2' # all 1e-3
            cut_small_female_savedir_2 = os.path.join(dir, female_data_name, mode, '')
            if not os.path.isdir(cut_small_female_savedir_2):
                os.makedirs(cut_small_female_savedir_2)
            plot_wav(small_mag_pad_2, file_number, '', cut_small_female_savedir_2)
            make_wav(small_mag_pad_2, file_number, '', cut_small_female_savedir_2)
            #female_infor_array.append(speaker_id)  # append speaker_id to list, 0
            #female_infor_array.append(file_number)  # append file_number to list, 1
            female_infor_array.append(origin_mag_pad)  # append spectrogram magnitude to list, 0
            #female_infor_array.append(origin_pha_pad)
            female_infor_array.append(small_mag_pad_1) # append cut small value spectrogram phase to list, 1
            female_infor_array.append(small_mag_pad_2)  # append cut small value spectrogram phase to list, 2
            female_final_array.append(female_infor_array)  # all infor append to final_array
            female_phase_array.append(origin_pha_pad)  # append spectrogram phase to list
            female_infor_array = []
        else:
            x, sr = librosa.load(read_file[k], sr=16000)  # to male, same as female
            (wav_check3, file_number) = os.path.split(read_file[k])
            (wav_check4, speaker_id) = os.path.split(wav_check3)
            file_number = file_number[:-4]
            compare_stft = librosa.stft(x, n_fft=nfft, hop_length=hop)
            output = my_dtw(refer_stft.T, compare_stft.T, len(refer_stft.T),len(compare_stft.T))  # using dtw, output is path
            print("k = {}, id = {}, male compare_stft len = {}, output len = {}".format(k,speaker_id,len(compare_stft.T), len(output)))
            final_output = compare_stft[:, output]  # compare source -> adapt path
            origin_mag, origin_pha = return_mag_pha(final_output)
            # pad_stft = zero_padding_complex(final_output, maxlen_stft)
            # mag_pad, pha_pad = return_mag_pha(pad_stft)
            cut_small_mag = cut_small_value(origin_mag)  # cut less than 1e-2 -> 1e-3
            # zero padding with origin mag, pha(complex type), small_value
            origin_mag_pad = zero_padding(origin_mag, maxlen_mag)
            origin_pha_pad = zero_padding_complex(origin_pha, maxlen_pha)
            small_mag_pad_1 = zero_padding(cut_small_mag, maxlen_mag)
            small_mag_pad_2 = cut_small_value(origin_mag_pad)
            # save original zero padding to plot, wav
            mode = 'origin_zero_padding_data'
            original_male_savedir = os.path.join(dir, male_data_name, mode, '')
            if not os.path.isdir(original_male_savedir):
                os.makedirs(original_male_savedir)
            plot_wav(final_output, file_number, '', original_male_savedir)
            make_wav(final_output, file_number, '', original_male_savedir)
            # save cut_small_value zero padding to plot, wav
            mode = 'cut_small_zero_padding_data_1'  # 1e-3, 0 also have
            cut_small_male_savedir_1 = os.path.join(dir, male_data_name, mode, '')
            if not os.path.isdir(cut_small_male_savedir_1):
                os.makedirs(cut_small_male_savedir_1)
            plot_wav(small_mag_pad_1, file_number, '', cut_small_male_savedir_1)
            make_wav(small_mag_pad_1, file_number, '', cut_small_male_savedir_1)
            # female_infor_array.append(speaker_id)  # append speaker_id to list, 0

            mode = 'cut_small_zero_padding_data_2'  # all 1e-3
            cut_small_male_savedir_2 = os.path.join(dir, male_data_name, mode, '')
            if not os.path.isdir(cut_small_male_savedir_2):
                os.makedirs(cut_small_male_savedir_2)
            plot_wav(small_mag_pad_2, file_number, '', cut_small_male_savedir_2)
            make_wav(small_mag_pad_2, file_number, '', cut_small_male_savedir_2)
            # female_infor_array.append(speaker_id)  # append speaker_id to list, 0
            # female_infor_array.append(file_number)  # append file_number to list, 1
            male_infor_array.append(origin_mag_pad)  # append spectrogram magnitude to list, 0
            #male_infor_array.append(origin_pha_pad)  # append spectrogram phase to list, 1
            male_infor_array.append(small_mag_pad_1)  # append cut small value spectrogram phase to list, 1
            male_infor_array.append(small_mag_pad_2)  # append cut small value spectrogram phase to list, 2
            male_final_array.append(male_infor_array)  # all infor append to final_array
            male_phase_array.append(origin_pha_pad)
            male_infor_array = []
        print("done in {} and speaker={}".format(k, speaker_id))
    np_female_final_array = np.asarray(female_final_array)
    #female_final_array = []
    np_male_final_array = np.asarray(male_final_array)

    np_female_phase_array = np.asarray(female_phase_array)
    np_male_phase_array = np.asarray(male_phase_array)
    #male_final_array = []
    #print('here ')

    mode = 'original_mag'
    original_female_mag = os.path.join(female_dir, mode)
    with open(original_female_mag, 'wb') as fp:
        pickle.dump(np_female_final_array[:,0], fp) # female mag, 0
    original_male_mag = os.path.join(male_dir, mode)
    with open(original_male_mag, 'wb') as fp:
        pickle.dump(np_male_final_array[:,0], fp) # male mag, 0

    mode = 'phase'
    female_pha = os.path.join(female_dir, mode)
    with open(female_pha, 'wb') as fp:
        pickle.dump(np_female_phase_array, fp) # female mag, 1
    male_pha = os.path.join(male_dir, mode)
    with open(male_pha, 'wb') as fp:
        pickle.dump(np_male_phase_array, fp)  # male mag, 1

    mode = 'cut_small_value_1'
    female_cut_small_value = os.path.join(female_dir, mode)
    with open(female_cut_small_value, 'wb') as fp:
        pickle.dump(np_female_final_array[:, 1], fp)  # male small value mag, 1
    male_cut_small_value = os.path.join(male_dir, mode)
    with open(male_cut_small_value, 'wb') as fp:
        pickle.dump(np_male_final_array[:, 1], fp)  # male small value mag, 1

    mode = 'cut_small_value_2'
    female_cut_small_value2 = os.path.join(female_dir, mode)
    with open(female_cut_small_value2, 'wb') as fp:
        pickle.dump(np_female_final_array[:, 2], fp)  # male small value mag, 2
    male_cut_small_value2 = os.path.join(male_dir, mode)
    with open(male_cut_small_value2, 'wb') as fp:
        pickle.dump(np_male_final_array[:, 2], fp)  # male small value mag, 2

    print('{}th dtw, mag, pha, small_value is done'.format(a))
print('all finish')
    #print('done??')




print('done')
# align Y to X




'''
# male_name = 'awb'
# female_name = 'slt'
male_name = 'masch0'
female_name = 'fbymh0'
country = 'kr'
# country = 'us'

wavfile_female = '/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/public/fbymh0/PBSG001.wav'
wavfile_male = '/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/public/masch0/PBSG001.wav'
# wavfile_female = '/mnt/junewoo/speech/cmu/for_transformer/woman_arctic_a0010.wav'
# wavfile_male = '/mnt/junewoo/speech/cmu/for_transformer/man_arctic_a0010.wav'
y_female, sr = librosa.load(wavfile_female, sr=16000)
y_male, sr = librosa.load(wavfile_male, sr=16000)
male_stft = librosa.stft(y_male, n_fft=nfft, hop_length=hop_length)
female_stft = librosa.stft(y_female, n_fft=nfft, hop_length=hop_length)

# using dtw to Y
y = my_dtw(male_stft.T, female_stft.T)
# align Y to X
female_stft = female_stft[:, y]
'''


'''original mag, phase'''
# real_original


''' ee
#male & female mag
with open(origin_male_mag, 'wb') as fp:
    pickle.dump(male_mag, fp)
origin_female_mag = dir+mode+female_name+name
with open(origin_female_mag, 'wb') as fp:
    pickle.dump(female_mag, fp)

#male & female phase
name = '_nfft='+str(nfft)+'hop='+str(hop_length)+'pha'
mode = 'ori_'
origin_male_phase = dir+mode+male_name+name
with open(origin_male_phase, 'wb') as fp:
    pickle.dump(male_phase, fp)
origin_female_phase = dir+mode+female_name+name
with open(origin_female_phase, 'wb') as fp:
    pickle.dump(female_phase, fp)

log10_male_mag = np.log10(cut_male_mag)
log10_female_mag = np.log10(cut_female_mag)

name = '_nfft='+str(nfft)+'hop='+str(hop_length)+'mag'
mode = 'log10_'
log10_male_mag_dir = dir+mode+male_name+name
with open(log10_male_mag_dir, 'wb') as fp:
    pickle.dump(log10_male_mag, fp)
log10_female_mag_dir = dir+mode+female_name+name
with open(log10_female_mag_dir, 'wb') as fp:
    pickle.dump(log10_female_mag, fp)
print("common log done")

# male & female ln log
ln_log_male_mag = np.log(cut_male_mag)
ln_log_female_mag = np.log(cut_female_mag)
mode = 'ln_'
lnlog_male_mag = dir+mode+male_name+name
with open(lnlog_male_mag, 'wb') as fp:
    pickle.dump(ln_log_male_mag, fp)
lnlog_female_mag = dir+mode+female_name+name
with open(lnlog_female_mag, 'wb') as fp:
    pickle.dump(ln_log_female_mag, fp)
print("ln log done")
'''


'''
# male & female mag exp log
e_log_male_mag = np.exp(male_mag)
e_log_female_mag = np.exp(female_mag)
name = '_nfft='+str(nfft)+'hop='+str(hop_length)+'mag'
mode = 'exp_'
exp_log_male_mag = dir+mode+male_name+name
with open(exp_log_male_mag, 'wb') as fp:
    pickle.dump(e_log_male_mag, fp)
exp_log_female_mag = dir+mode+female_name+name
with open(exp_log_female_mag, 'wb') as fp:
    pickle.dump(e_log_female_mag, fp)

test1_male = np.log(e_log_male_mag)

# male & female ln log
ln_log_male_mag = np.log(male_mag)
ln_log_female_mag = np.log(female_mag)
mode = 'ln_'
lnlog_male_mag = dir+mode+male_name+name
with open(lnlog_male_mag, 'wb') as fp:
    pickle.dump(ln_log_male_mag, fp)
lnlog_female_mag = dir+mode+female_name+name
with open(lnlog_female_mag, 'wb') as fp:
    pickle.dump(ln_log_female_mag, fp)

test_male = np.exp(ln_log_male_mag)


# norm male & female
mode = 'norm_'
M_max, M_min = male_mag.max(), male_mag.min()
norm_male_mag = (male_mag - M_min)/(M_max - M_min)
norm_male = dir+mode+male_name+name
with open(norm_male, 'wb') as fp:
    pickle.dump(norm_male_mag, fp)

F_max, F_min = female_mag.max(), female_mag.min()
norm_female_mag = (female_mag - F_min)/(F_max - F_min)
norm_female = dir+mode+female_name+name
with open(norm_female, 'wb') as fp:
    pickle.dump(norm_female_mag, fp)

print("M_max {} M_min {} F_max {} F_min {}".format(M_max, M_min, F_max, M_min))
denorm = (norm_female_mag * (F_max - F_min)) + (F_min)
a = []

#a.append(M_max,M_min,F_max,F_min)
a.append(M_max)
a.append(M_min)
a.append(F_max)
a.append(F_min)
print(np.shape(a))
print(a)
save_dir = dir+'max_min'+'nfft='+str(nfft)+',hop='+str(hop_length)
with open(save_dir, 'wb') as fp:
    pickle.dump(a, fp)

'''