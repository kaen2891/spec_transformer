#-*- coding: utf-8 -*-

import numpy as np
import os
from glob import glob
import shutil
overlap_dir = '/mnt/junewoo/speech/ETRI/overlap_data/'

#all_txt = glob('/mnt/junewoo/speech/ETRI/wav_korean/*/*.txt')
#all_txt = glob('/mnt/junewoo/speech/ETRI/wav_korean/*/*.txt')


def check_overlap(glob_list):
    tmp_name = []
    tmp_name2 = []
    tmp_name3 = []
    for i in range(len(glob_list)):
        with open(glob_list[i], 'rt', encoding='UTF8') as f:
            lines = f.read()
            # lines = f.read().decode('cp949')
            lines = lines.strip()
            print(lines)
            if lines not in tmp_name:
                tmp_name.append(lines)
            else:
                tmp_name2.append(lines)  # 중복되는 내용 삽입
                tmp_name3.append(glob_list[i])  # 중복되는 내용의 directory 삽입
    # print("총 개수 : {}이고, 중복되지 않은 것은 {}개, 중복되는 것은 {}개 이다.".format(len(all_txt), len(tmp_name), len(tmp_name2)))
    # print("중복되는 내용은 {}이다.".format(tmp_name2))
    # print("중복되는 파일의 경로는 {} 이다".format(tmp_name3))
    return tmp_name3

def merge_overlap_to_list(glob_list, overlap_file):
    overlap_list = []
    overlap_word = []
    cnt = 0
    # 모든 text 중 중복되는 것 list에 넣기
    for i in range(len(glob_list)):
        with open(glob_list[i], 'rt', encoding='UTF8') as f:
            lines = f.read()
            # lines = f.read().decode('cp949')
            lines = lines.strip()
            for ii in range(len(overlap_file)):
                with open(overlap_file[ii], 'rt', encoding='UTF8') as f2:  # 중복되는 디렉토리의 내용
                    over_lines = f2.read()
                    over_lines = over_lines.strip()
                    if lines == over_lines:
                        overlap_list.append(glob_list[i])
                        overlap_word.append(lines)
                        cnt += 1
                        break
        # print("{}번째 작업 끝".format(i))
    #print("overlab list : ", overlap_list)
    #print("overlab word : ", overlap_word)
    final_ylist = []

    for i in range(len(overlap_list)):  # 4개, i=0
        sum = 0
        sum_i = 0
        y_list = []
        for y in range(i + 1, cnt):  # 0, 3까지
            # print("i = {}, y = {}".format(i, y))
            # for xx in range(len(final_ylist)+1):
            # print("xx = {} and y = {} and len(final_ylist) = {}".format(xx, y, len(final_ylist)))
            # print(final_ylist[xx])
            # if y-1 not in final_ylist[xx]:
            if overlap_word[i] == overlap_word[y]:
                #
                if len(final_ylist) == 0:
                    if i not in y_list:
                        y_list.append(i)
                    y_list.append(y)
                    # final_ylist.append(y_list)
                    # print("final ylist 는:",final_ylist)
                    continue
                # print("final 길이:", len(final_ylist)) #1
                for xx in range(len(final_ylist)):  # 0,1
                    # print("xx = {} 이고 len(final_ylist)는 {}이고 y = {} 이고 i = {} 이다.".format(xx, len(final_ylist), y, i))
                    if i in final_ylist[xx]:
                        sum_i += 1
                    if y in final_ylist[xx]:
                        sum += 1
                if sum_i == 0 and i not in y_list:
                    y_list.append(i)
                if sum == 0:
                    y_list.append(y)
        if len(y_list) != 0:
            final_ylist.append(y_list)

def do_copy_ETRI_child(folder_name, final_ylist, overlap_list):
    save_dir = os.path.join(overlap_dir, folder_name)  # save_dir:/mnt/junewoo/speech/ETRI/overlap_data/child/
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 없을 경우 자동 생성

    count_number = 0
    for i in range(len(final_ylist)):  # 2
        count_number += 1
        print("{} 번째 작업".format(i))
        # print(final_ylist[i]) # 00
        for j in final_ylist[i]:
            # folder number
            cnt = str(count_number) + '/'
            save_dir_with_num = os.path.join(save_dir, cnt)
            if not os.path.exists(save_dir_with_num):
                os.makedirs(save_dir_with_num)
            # print("save_dir_with_num is :", save_dir_with_num)
            # file_id extract
            (dir, file_id) = os.path.split(overlap_list[j])
            # print("j = {}, dir = {}, file_id = {}".format(j, dir, file_id))
            (dir_dir, dir_id) = os.path.split(dir)
            # print("j = {}, dir_dir = {}, dir_id = {}".format(j, dir_dir, dir_id))
            (dir_dir_dir, dir_dir_id) = os.path.split(dir_dir)
            # print("j = {}, dir_dir_dir = {}, dir_dir_id = {}".format(j, dir_dir_dir, dir_dir_id))

            txt_save_dir = os.path.join(save_dir_with_num, file_id)
            # print("txt_save_dir", txt_save_dir)

            wave_id = file_id[:-8] + '_OFC' + file_id[8:12] + '.wav'
            wave_id = wave_id.replace('T', 'G')
            # print("wave_id is :", wave_id)
            # print(wave_id)
            # wave_id = file_id[:-4] + '.wav'

            new_name = file_id[:8] + '_OFC/'
            new_name = new_name.replace('T', 'G')
            # print(new_name)
            wav_file = os.path.join(dir_dir, new_name, wave_id)
            # print(wav_file)
            wav_save_dir = os.path.join(save_dir_with_num, wave_id)
            shutil.copy(overlap_list[j], txt_save_dir)  # txt copy
            shutil.copy(wav_file, wav_save_dir)
            print("{}, {} is done".format(txt_save_dir, wav_save_dir))

    finish_command = "Job Finish"
    return finish_command

def main():
    all_txt = glob(
    '/mnt/junewoo/speech/ETRI/child_speech/*/*/*.txt')  # /mnt/junewoo/speech/ETRI/child_speech/0001_M1/TXT0001_M1/
    print("Text 파일 총 길이: ",len(all_txt))

    overlap_file = check_overlap(all_txt)
    overlap_list, final_ylist = merge_overlap_to_list(all_txt,overlap_file)

    folder_name = 'child/'

    finish_cmd = do_copy_ETRI_child(folder_name, final_ylist, overlap_list)

    print(finish_cmd)



if __name__ == '__main__':
    main()







# Test
'''
kount = 0

for g in range(len(all_txt_array)):
    if all_txt_array[g] == '배':
        kount +=1
print("조개 개수는 :",kount)
'''