import numpy as np
from glob import glob
#import sys
import wave
import os
import shutil

def convert2wav_2(glob_list, save_dir):
    for i in range(len(glob_list)):
    #for i in range(10):
        (dir, file_id) = os.path.split(glob_list[i]) # dir + filename.pcm
        print('dir is {} and file_id is {}'.format(dir, file_id))
        (dir_dir, dir_id) = os.path.split(dir)
        print('dir_dir is {} and dir_id is {}'.format(dir_dir, dir_id))
        (dir_dir_dir, dir_dir_id) = os.path.split(dir_dir)
        print('dir_dir_dir is {} and dir_dir_id is {}'.format(dir_dir_dir, dir_dir_id))


        with open(glob_list[i], 'rb') as pcmfile:
            pcmdata = pcmfile.read()
        cut_pcm = file_id[:-4]
        print("cut_pcm is :", cut_pcm)
        new_dir = os.path.join(save_dir, dir_dir_id, dir_id)
        print("new dir is ",new_dir)

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        final_dir = os.path.join(new_dir, cut_pcm)
        
        with wave.open(final_dir + '.wav', 'wb') as wavfile:
            wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
            wavfile.writeframes(pcmdata)


def move_txtfile_2(glob_list, save_dir):
    for i in range(len(glob_list)):
        (dir, file_id) = os.path.split(glob_list[i])  # dir + filename.pcm
        print('dir is {} and file_id is {}'.format(dir, file_id))
        (dir_dir, dir_id) = os.path.split(dir)
        print('dir_dir is {} and dir_id is {}'.format(dir_dir, dir_id))
        (dir_dir_dir, dir_dir_id) = os.path.split(dir_dir)
        print('dir_dir_dir is {} and dir_dir_id is {}'.format(dir_dir_dir, dir_dir_id))

        new_dir = os.path.join(save_dir, dir_dir_id, dir_id)
        #print(new_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        final_dir = os.path.join(new_dir, file_id)
        #print(final_dir)
        print("final_dir is :", final_dir)
        shutil.copy(glob_list[i], final_dir)



def convert2wav(glob_list, save_dir):
    for i in range(len(glob_list)):
        (dir, file_id) = os.path.split(glob_list[i]) # dir + filename.pcm
        (dir_dir, dir_id) = os.path.split(dir)
        with open(glob_list[i], 'rb') as pcmfile:
            pcmdata = pcmfile.read()
        cut_pcm = file_id[:-4]
        new_dir = os.path.join(save_dir, dir_id)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        final_dir = os.path.join(new_dir, cut_pcm)
        with wave.open(final_dir + '.wav', 'wb') as wavfile:
            wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
            wavfile.writeframes(pcmdata)



def move_txtfile(glob_list, save_dir):
    for i in range(len(glob_list)):
        (dir, file_id) = os.path.split(glob_list[i])
        (dir_dir, dir_id) = os.path.split(dir)
        #print(dir_id)
        new_dir = os.path.join(save_dir, dir_id)
        #print(new_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        final_dir = os.path.join(new_dir, file_id)
        #print(final_dir)
        shutil.copy(glob_list[i], final_dir)

korean_dir = '/mnt/junewoo/speech/ETRI/wav_korean/'
english_dir = '/mnt/junewoo/speech/ETRI/wav_english/'

english_pcm = glob('/mnt/junewoo/speech/ETRI/download_4_Korean_English_by_Korean_part1/*/*.pcm')
english_txt = glob('/mnt/junewoo/speech/ETRI/download_4_Korean_English_by_Korean_part1/*/*.txt')

korean_pcm = glob('/mnt/junewoo/speech/ETRI/download_4_Korean_English_by_Korean_part2/*/*.pcm')
korean_txt = glob('/mnt/junewoo/speech/ETRI/download_4_Korean_English_by_Korean_part2/*/*.txt')

child_dir = '/mnt/junewoo/speech/ETRI/child_speech/'

child_pcm = glob('/mnt/junewoo/speech/ETRI/download_5_child_Korean/*/*/*.pcm')
child_txt = glob('/mnt/junewoo/speech/ETRI/download_5_child_Korean/*/*/*.txt')

convert2wav(english_pcm, english_dir) # done
move_txtfile(english_txt, english_dir) # done
#convert2wav_2(child_pcm , child_dir)
#move_txtfile_2(child_txt, child_dir)



'''
file_dir = '/mnt/junewoo/speech/ETRI/download_4_Korean_English_by_Korean_part1/I4F07443LJA0/SW201105ETRNI4F07443LJA0001.pcm'
file_dir2 = '/mnt/junewoo/speech/ETRI/download_4_Korean_English_by_Korean_part2/GSM02544LHS0/SW201105KTRNGSM02544LHS0001.pcm'
(dir, file_id) = os.path.split(file_dir2)
print(dir)
print(file_id)
with open(file_dir2, 'rb') as pcmfile:
    pcmdata = pcmfile.read()
new_wav = file_dir2[:-4]
print(new_wav)
(new, new_id) = os.path.split(new_wav)
print(new)
print(new_id)
with wave.open(new_wav+'.wav', 'wb') as wavfile:
    wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
    wavfile.writeframes(pcmdata)
'''