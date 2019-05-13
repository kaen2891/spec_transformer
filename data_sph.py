from sphfile import SPHFile
from glob import glob
import os
from random import shuffle
import shutil

#done
def step1_to_convert_wav(glob_list, Subject_name):
    print("input length is :", len(glob_list))
    for i in range(len(glob_list)):
        (dir, file_id) = os.path.split(glob_list[i])  # dir + filename.pcm
        (dir_dir, speaker_id) = os.path.split(dir)
        new_dir = '/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/'
        file_id = file_id[:-3]+'wav'
        sph = SPHFile(glob_list[i])
        save_dir1 = os.path.join(new_dir,Subject_name,speaker_id)
        if not os.path.exists(save_dir1):
            os.makedirs(save_dir1)
        current_path = os.getcwd()
        print("1st current path is: ",current_path)
        os.chdir(save_dir1)
        sph.write_wav(file_id)
        current_path2 = os.getcwd()
        print("save file name is :", file_id)
        print("after save, current path is :",current_path2)

def step2_to_convert_wav(glob_list, Subject_name):
    print("input length is :", len(glob_list))
    # shuffle(glob_list)
    for i in range(len(glob_list)):
        (dir, file_id) = os.path.split(glob_list[i])  # dir + filename.pcm
        (dir_dir, pbw_number) = os.path.split(dir)
        (dir_dir_dir, speaker_number) = os.path.split(dir_dir)
        # print("dir : {}, file_id : {} , dir_dir : {}, speaker_id : {}, dir_dir_dir : {}, speaker_number : {}".format(dir, file_id, dir_dir, pbw_number, dir_dir_dir, speaker_number))
        new_dir = '/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/'
        middle_dir = os.path.join(new_dir, Subject_name)
        save_dir = os.path.join(middle_dir, speaker_number)
        file_id = file_id[:-3]+'wav'
        # print("file id is:", file_id)
        # print("new dir is :", new_dir)
        # print("save dir is :", save_dir)
        sph = SPHFile(glob_list[i])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        current_path = os.getcwd()

        # print("1st current path is: ",current_path)
        os.chdir(save_dir)
        sph.write_wav(file_id)
        current_path2 = os.getcwd()
        # print("save file name is :", file_id)
        # print("after save, current path is :",current_path2)
        print("{}th WAV file is saved".format(i))


def common_data(glob_list, subject_name):
    #shuffle(glob_list)
    for i in range(len(glob_list)):
        (dir, file_id) = os.path.split(glob_list[i])  # dir + filename.pcm
        (dir_dir, speaker_id) = os.path.split(dir)
        # print("file_id is {} speaker_id is {}".format(file_id, speaker_id))
        public_dir = '/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/'
        new_dir = os.path.join(public_dir, subject_name)
        save_dir = os.path.join(new_dir, speaker_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        shutil.copy(glob_list[i], save_dir)

# def divide_dataset(glob_list, subject_name):


# PBS_data = glob('/mnt/junewoo/speech/KAIST_wav/Korean_Voice_DB/PBS/KKWON/*/*.WV1')
PBW_data = glob('/mnt/junewoo/speech/KAIST_wav/Korean_Voice_DB/Pbw/KKWON/PBW/*/*/*.WAV')
NAR_data = glob('/mnt/junewoo/speech/KAIST_wav/Korean_Voice_DB/Pbw/KKWON/NAR/*/*.WAV')
DIG_data = glob('/mnt/junewoo/speech/KAIST_wav/Korean_Voice_DB/Pbw/KKWON/DIG/*/*.WAV')
CDIG_data = glob('/mnt/junewoo/speech/KAIST_wav/Korean_Voice_DB/Pbw/KKWON/CDIG/*/*.WAV')
# print(len(NAR_data))
#
Subject_name = 'CDIG'
# step1_to_convert_wav(DIG_data,Subject_name)
# step2_to_convert_wav(NAR_data, Subject_name)

prepro_PBS_data = glob('/mnt/junewoo/speech/KAIST_wav/new_Korean_Voice_DB/PBS/*/*G*.wav')

Subject_name = 'public'
common_data(prepro_PBS_data, Subject_name)



#
# sph =SPHFile(
#     '/mnt/junewoo/speech/KAIST_wav/Korean_Voice_DB/Pbw/KKWON/PBW/FGBS0/PBW1/PBW1147.WAV'
# )
# # Note that the following loads the whole file into ram
# print(sph.format)
# # write out a wav file with content from 111.29 to 123.57 seconds
# sph.write_wav('test.wav')
