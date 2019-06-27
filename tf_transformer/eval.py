import tensorflow as tf
import argparse
import os
import re
import sys
import model as ml
import pickle
import data
import numpy as np
from configs import DEFINES
import librosa
import librosa.display
import matplotlib.pyplot as plt
import mel_to_inverse
from datetime import datetime

RESULT_FIGURE = './result_figure/'
RESULT_WAV = './result_wav/'
parser = argparse.ArgumentParser()
new_args = vars(parser.parse_args())
parser.add_argument('--batch_size', default='5', help='batch size')
parser.add_argument('--train_steps', default='20000', help='train_steps')
parser.add_argument('--checkpoint', default='/mnt/junewoo/workspace/transform/data_out/check_point4', help='Path to model checkpoint')

frame_length = 0.32
frame_stride = 0.1  # length / 0.1
fs = 16000

def make_wav(mel, input_nfft, input_stride, sr, save_dir, file_save_name):
    test = mel_to_inverse.Convert2Wav(wav=None, mel_spec=mel, frame_length=input_nfft, frame_stride=input_stride,
                                      sr=sr, save_dir=save_dir, save_name=file_save_name)
    play = test.mel2wav()

def plot_melspec(name, predict_result):
    now = datetime.now()
    frame_length = 0.32
    frame_stride = 0.1  # length / 0.1
    fs = 16000
    input_stride = int(round(fs * frame_stride))

    librosa.display.specshow(librosa.power_to_db(predict_result, ref=np.max), y_axis='mel', sr=fs,
                             hop_length=input_stride,
                             x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig(RESULT_FIGURE + '/'+'{}-{}-{}'.format(now.day, now.hour, now.minute)+name+'transformer_Mel-Spectrogram.png')
    plt.show()
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    with open('/mnt/junewoo/workspace/transform/test_head/x_train', 'rb') as file:
        X_train = pickle.load(file)

    # log mel
    # with open('/mnt/junewoo/workspace/transform/log_mel/log_X', 'rb') as file: # log
    #     X_train = pickle.load(file)

    # log s^2 mel
    # with open('/mnt/junewoo/workspace/transform/log_mel/log_s2_X', 'rb') as file: # log s^2
    #     X_train = pickle.load(file)

    # X_train = np.asarray(X_train)
    # now_test = X_train[0]
    # show_test = np.asarray(now_test)
    # show_test2 = np.transpose(show_test, (1, 0))
    # transpose_spec = []
    # for i in range(len(X_train)):
    #     spec = np.transpose(X_train[i], (1, 0))
    #     transpose_spec.append(spec)
    # transpose_spec = np.asarray(transpose_spec)

    X_train = X_train[0]
    X_train = X_train[np.newaxis]
    # Y_train = Y_train[0]
    # Y_train = Y_train[np.newaxis]


    predic_input_enc = np.asarray(X_train)
    print(np.shape(predic_input_enc))
    predic_output_dec = np.zeros((1, 60, 128))
    predic_output_dec = np.float32(predic_output_dec)
    predic_target_dec = np.zeros((1, 60, 128))
    predic_target_dec = np.float32(predic_target_dec)
    # print(predic_input_enc.dtype, predic_output_dec.dtype, predic_target_dec.dtype)
    # print(type(predic_input_enc), type(predic_output_dec), type(predic_target_dec))

    classifier = tf.estimator.Estimator(
        model_fn=ml.Model,  # 모델 등록한다.
        model_dir=DEFINES.check_point_path,
        # model_dir=new_args['checkpoint'],  # 체크포인트 위치 등록한다.
        params={  # 모델 쪽으로 파라메터 전달한다.
            'embedding_size': DEFINES.embedding_size,
            'model_hidden_size': DEFINES.model_hidden_size,  # 가중치 크기 설정한다.
            'ffn_hidden_size': DEFINES.ffn_hidden_size,
            'attention_head_size': DEFINES.attention_head_size,
            'learning_rate': DEFINES.learning_rate,  # 학습율 설정한다.
            'layer_size': DEFINES.layer_size,
            'max_sequence_length': DEFINES.max_sequence_length,
        })
    predictions = classifier.predict(
        input_fn= lambda: data.eval_input_fn(predic_input_enc, predic_output_dec, predic_target_dec, 1))
        # input_fn=lambda: data.eval_input_fn(predic_input_enc, predic_output_dec, predic_target_dec, 1))
    print("predictions",predictions)
    print("predictions type",type(predictions))
    print("predictions shape",np.shape(predictions))
    a = []
    for prediction in predictions:
        a.append(prediction)

    # result = [i['logits'] for i in a]
    predict_input_enc = a
    predict_input_enc = predict_input_enc[0]
    # predict_output_dec = result[1]
    # predict_target_dec = result[2]

    # 체크해야함
    final_predict_input_enc = np.transpose(predict_input_enc, (1, 0))
    # final_predict_output_dec = np.transpose(predict_output_dec, (1, 0))
    # final_predict_target_dec = np.transpose(predict_target_dec, (1, 0))

    now = datetime.now()
    #input enc
    # input_name = 'X_train'
    input_name = 'X_train'
    # input_name = 'X_train_log_s^2'
    # name = input_name+'predict_input_enc_log_s^2'
    name = input_name+' predict_input_enc'
    plot_melspec(name, final_predict_input_enc)
    # make_wav(final_predict_input_enc, frame_length, frame_stride, fs, RESULT_WAV, input_name+'{}-{}-{}predict_inputenc_wav_log'.format(now.day, now.hour, now.minute))
    make_wav(final_predict_input_enc, frame_length, frame_stride, fs, RESULT_WAV,
             name + '{}-{}-{} '.format(now.day, now.hour, now.minute))

    name = input_name+'predict_input_enc_Reverse_to_origin'
    # reverse_predict_input_enc = librosa.core.db_to_amplitude(final_predict_input_enc) # log s^2
    reverse_predict_input_enc = librosa.core.power_to_db(final_predict_input_enc) # log
    plot_melspec(name, reverse_predict_input_enc)
    make_wav(reverse_predict_input_enc, frame_length, frame_stride, fs, RESULT_WAV, name+'{}-{}-{} '.format(now.day, now.hour, now.minute))


    #output_dec
    # name = 'predict_output_dec_'
    # plot_melspec(name, final_predict_output_dec)
    # make_wav(final_predict_input_enc, frame_length, frame_stride, fs, RESULT_WAV, 'predict_outputdec_wav_log')
    #
    # name = 'predict_output_dec_reverse'
    # reverse_predict_output_dec = librosa.core.db_to_amplitude(final_predict_output_dec)
    # plot_melspec(name, reverse_predict_output_dec)
    # make_wav(reverse_predict_output_dec, frame_length, frame_stride, fs, RESULT_WAV, 'predict_outputdec_wav_origin')
    #
    # name = 'predict_target_dec_'
    # plot_melspec(name, final_predict_target_dec)
    # make_wav(final_predict_target_dec, frame_length, frame_stride, fs, RESULT_WAV, 'predict_targetdec_wav_log')
    #
    # name = 'predict_target_dec_reverse'
    # reverse_predict_target_dec = librosa.core.db_to_amplitude(final_predict_target_dec)
    # plot_melspec(name, reverse_predict_target_dec)
    # make_wav(reverse_predict_target_dec, frame_length, frame_stride, fs, RESULT_WAV, 'predict_targetdec_wav_origin')

    # plot test result
    # with open('/mnt/junewoo/workspace/transform/test_head/y_test', 'rb') as file:
    #     Y_test = pickle.load(file)
    with open('/mnt/junewoo/workspace/transform/test_head/y_train', 'rb') as file:
        Y_train = pickle.load(file)
    # log mel
    # with open('/mnt/junewoo/workspace/transform/log_mel/log_Y', 'rb') as file: # log
    #     Y_train = pickle.load(file)
    # with open('/mnt/junewoo/workspace/transform/log_mel/log_s2_Y', 'rb') as file: # log s^2
    #     Y_train = pickle.load(file)
    # test_set = Y_train[0]
    # final_test = np.transpose(test_set, (1, 0))

    # name = 'original_log_s^2'
    # name = 'original'
    # plot_melspec(name, final_test)
    # make_wav(final_test, frame_length, frame_stride, fs, RESULT_WAV, '{}-{}-{}y'.format(now.day, now.hour, now.minute))

    # name = 'original_reverse_log_s^2'
    # origin_reverse = librosa.core.db_to_amplitude(final_test)
    # plot_melspec(name, origin_reverse)
    # make_wav(origin_reverse, frame_length, frame_stride, fs, RESULT_WAV, '{}-{}-{}y'.format(now.day, now.hour, now.minute))


if __name__ == '__main__':
  main()