import tensorflow as tf
import model as ml
import data
import numpy as np
import os
import sys
import pickle
from configs import DEFINES
import matplotlib.pyplot as plt

tf.reset_default_graph()
os.environ['CUDA_VISIBLE_DEVICES']='6'
# DATA_OUT_PATH = './data_out/'


def main(self):
    # data_out_path = os.path.join(os.getcwd(), DATA_OUT_PATH)
    # os.makedirs(data_out_path, exist_ok=True)
    # # 데이터를 통한 사전 구성 한다.
    # char2idx, idx2char, vocabulary_length = data.load_vocabulary()

    with open('/mnt/junewoo/workspace/transform/test_head/x_train', 'rb') as file:
        X_train = pickle.load(file)
    with open('/mnt/junewoo/workspace/transform/test_head/y_train', 'rb') as file:
        Y_train = pickle.load(file)
    with open('/mnt/junewoo/workspace/transform/test_head/x_val', 'rb') as file:
        X_val = pickle.load(file)
    with open('/mnt/junewoo/workspace/transform/test_head/y_val', 'rb') as file:
        Y_val = pickle.load(file)
    with open('/mnt/junewoo/workspace/transform/test_head/x_test', 'rb') as file:
        X_test = pickle.load(file)
    with open('/mnt/junewoo/workspace/transform/test_head/y_test', 'rb') as file:
        Y_test = pickle.load(file)
    # print("X_train 길이:",len(X_train))
    # print(np.shape(X_train[0]))
    # print(len(X_train))
    # 훈련 데이터와 테스트 데이터를 가져온다.
    # train_input, train_label, test_input, test_label = X_train, Y_train, X_test, Y_test

    # 훈련셋 인코딩 만드는 부분이다.
    train_input_enc, train_input_enc_length = np.asarray(X_train), 60

    # 훈련셋 디코딩 입력 부분 만드는 부분이다.
    train_output_dec, train_output_dec_length = np.asarray(Y_train), 60

    # 훈련셋 디코딩 출력 부분 만드는 부분이다.
    train_target_dec = np.asarray(Y_train)
    #
    # 평가셋 인코딩 만드는 부분이다.
    eval_input_enc, eval_input_enc_length = np.asarray(X_test), 60
    # 평가셋 인코딩 만드는 부분이다.
    eval_output_dec, eval_output_dec_length = np.asarray(Y_test), 60
    # 평가셋 인코딩 만드는 부분이다.
    eval_target_dec = np.asarray(Y_test)

    # 현재 경로'./'에 현재 경로 하부에
    # 체크 포인트를 저장한 디렉토리를 설정한다.
    check_point_path = os.path.join(os.getcwd(), DEFINES.check_point_path)
    # 디렉토리를 만드는 함수이며 두번째 인자 exist_ok가
    # True이면 디렉토리가 이미 존재해도 OSError가
    # 발생하지 않는다.
    # exist_ok가 False이면 이미 존재하면
    # OSError가 발생한다.
    os.makedirs(check_point_path, exist_ok=True)

    # print(train_input_enc)
    print(np.shape(train_input_enc))
    print("type:",type(train_input_enc))
    # print("")
    total_loss = []
    # 에스티메이터 구성한다.
    classifier = tf.estimator.Estimator(
        model_fn=ml.Model,  # 모델 등록한다.
        model_dir=DEFINES.check_point_path,  # 체크포인트 위치 등록한다.
        params={  # 모델 쪽으로 파라메터 전달한다.
            'embedding_size': DEFINES.embedding_size,
            'model_hidden_size': DEFINES.model_hidden_size,  # 가중치 크기 설정한다.
            'ffn_hidden_size': DEFINES.ffn_hidden_size,
            'attention_head_size': DEFINES.attention_head_size,
            'learning_rate': DEFINES.learning_rate,  # 학습율 설정한다.
            'layer_size': DEFINES.layer_size,
            'max_sequence_length': DEFINES.max_sequence_length,
            'dropout_width': DEFINES.dropout_width
            # 'dropout_rate': DEFINES.drop
            # 'vocabulary_length': 60
            # 'xavier_initializer': DEFINES.xavier_initializer
        })
    # print("input_enc type {} output_dec type {} target_dec type {}".format(type(train_input_enc), type(train_output_dec), type(train_target_dec)))
    # 학습 실행
    classifier.train(input_fn=lambda: data.train_input_fn(
        train_input_enc, train_output_dec, train_target_dec, DEFINES.batch_size), steps=DEFINES.train_steps)

    # eval_loss = classifier.evaluate(input_fn= lambda: data.eval_input_fn(eval_input_enc, eval_output_dec, eval_target_dec, DEFINES.batch_size))
    # total_loss.append(eval_loss)
    #
    # plt.plot(2000, total_loss, 'b-', label='loss')
    # plt.title('L2 loss')
    # plt.xlabel('$Y_{pred}$', fontsize=15)
    # plt.ylabel('$Y_{true}$', fontsize=15)
    # plt.savefig('transformer_loss.png')
    # plt.show()



    # eval_result = classifier.evaluate(input_fn=lambda: data.eval_input_fn(
    #     eval_input_enc, eval_output_dec, eval_target_dec, DEFINES.batch_size))

    # print('\nEVAL set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    #

    #
    # # # 테스트용 데이터 만드는 부분이다.
    # # # 인코딩 부분 만든다.
    # predic_input_enc = np.asarray(X_val)
    # # # 학습 과정이 아니므로 디코딩 입력은
    # # # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    # predic_output_dec = np.zeros((3, 60, 128))
    # # # 학습 과정이 아니므로 디코딩 출력 부분도
    # # # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    # predic_target_dec = np.zeros((3, 60, 128))
    # #
    # predictions = classifier.predict(
    #     input_fn=lambda: data.eval_input_fn(predic_input_enc, predic_output_dec, predic_target_dec, 1))

    #
    # answer, finished = data.pred_next_string(predictions, idx2char)
    #
    # # 예측한 값을 인지 할 수 있도록
    # # 텍스트로 변경하는 부분이다.
    # print("answer: ", answer)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

tf.logging.set_verbosity