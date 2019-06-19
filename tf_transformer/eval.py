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
from datetime import datetime

RESULT_FIGURE = './result_figure/'
parser = argparse.ArgumentParser()
new_args = vars(parser.parse_args())
parser.add_argument('--batch_size', default='5', help='batch size')
parser.add_argument('--train_steps', default='20000', help='train_steps')
parser.add_argument('--checkpoint', default='/mnt/junewoo/workspace/transform/data_out/check_point4', help='Path to model checkpoint')

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    with open('/mnt/junewoo/workspace/transform/test_head/x_train', 'rb') as file:
        X_train = pickle.load(file)
    with open('/mnt/junewoo/workspace/transform/test_head/x_test', 'rb') as file:
        X_test = pickle.load(file)

    predic_input_enc = np.asarray(X_train)
    print(np.shape(predic_input_enc))
    predic_output_dec = np.zeros((7, 60, 128))
    predic_output_dec = np.float32(predic_output_dec)
    predic_target_dec = np.zeros((7, 60, 128))
    predic_target_dec = np.float32(predic_target_dec)
    print(predic_input_enc.dtype, predic_output_dec.dtype, predic_target_dec.dtype)
    print(type(predic_input_enc), type(predic_output_dec), type(predic_target_dec))

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

    result = [i['logits'] for i in a]
    predict_input_enc = result[0]
    predict_output_dec = result[1]
    predict_target_dec = result[2]

    # 체크해야함
    final_predict_input_enc = np.transpose(predict_input_enc, (1, 0))
    final_predict_output_dec = np.transpose(predict_output_dec, (1, 0))
    final_predict_target_dec = np.transpose(predict_target_dec, (1, 0))

    now = datetime.now()
    #input enc
    name = 'predict_input_enc_'
    plot_melspec(name, final_predict_input_enc)
    #output_dec
    name = 'predict_output_dec_'
    plot_melspec(name, final_predict_output_dec)

    name = 'predict_target_dec_'
    plot_melspec(name, final_predict_target_dec)

    # plot test result
    # with open('/mnt/junewoo/workspace/transform/test_head/y_test', 'rb') as file:
    #     Y_test = pickle.load(file)
    with open('/mnt/junewoo/workspace/transform/test_head/x_train', 'rb') as file:
        Y_train = pickle.load(file)
    test_set = Y_train[0]
    final_test = np.transpose(test_set, (1, 0))

    name = 'original_'
    plot_melspec(name, final_test)


if __name__ == '__main__':
  main()