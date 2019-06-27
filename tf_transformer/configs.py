# -*- coding: utf-8 -*-
import tensorflow as tf

tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size')  # 배치 크기
tf.app.flags.DEFINE_integer('train_steps', 100000, 'train steps')  # 학습 에포크
tf.app.flags.DEFINE_float('dropout_width', 0.1, 'dropout width')  # 드롭아웃 크기
tf.app.flags.DEFINE_integer('embedding_size', 128, 'embedding size')  # 가중치 크기 # 논문 512 사용
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')  # 학습률
tf.app.flags.DEFINE_integer('shuffle_seek', 1000, 'shuffle random seek')  # 셔플 시드값
tf.app.flags.DEFINE_integer('max_sequence_length', 60, 'max sequence length')  # 시퀀스 길이
tf.app.flags.DEFINE_integer('model_hidden_size', 128, 'model weights size')  # 모델 가중치 크기
tf.app.flags.DEFINE_integer('ffn_hidden_size', 512, 'ffn weights size')  # ffn 가중치 크기
tf.app.flags.DEFINE_integer('attention_head_size', 4, 'attn head size')  # 멀티 헤드 크기
# tf.app.flags.DEFINE_integer('attention_head_size', 16, 'attn head size')  # 멀티 헤드 크기
tf.app.flags.DEFINE_integer('layer_size', 3, 'layer size')  # layer 크기
# tf.app.flags.DEFINE_integer('layer_size', 12, 'layer size')  # layer 크기
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point', 'check point path')  # 체크 포인트 위치 (20190606)
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point2', 'check point path')  # 체크 포인트 위치 (20190607 09시 전)
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point3', 'check point path')  # 체크 포인트 위치 (20190607 09시부터~)
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point4', 'check point path')  # 체크 포인트 위치 (20190624 16시부터~), 10만번 train
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point5', 'check point path')  # 체크 포인트 위치 (20190618 18시부터~), 10만번 train
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point6', 'check point path')  # 체크 포인트 위치 (20190618 18시부터~)
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point7', 'check point path')  # 체크 포인트 위치 (20190619 16시부터~) loop_count부분
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point8', 'check point path')  # 체크 포인트 위치 (20190620 16시부터~) log mel
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point9', 'check point path')  # 체크 포인트 위치 (20190621 18시부터~) log s^2 mel + 10만번
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point10', 'check point path')  # 체크 포인트 위치 (20190621 18시부터~) log mel + 10만번
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point11', 'check point path')  # 체크 포인트 위치 (20190622 16시부터~) log s^2 mel, 10만번, multihead 개수 16
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point12', 'check point path')  # 체크 포인트 위치 (20190622 16시부터~) log mel, 10만번, multihead 개수 16
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point13', 'check point path')  # 체크 포인트 위치 (20190622 16시부터~) log s^2 mel, 10만번, multihead 개수 16, layer 개수 12
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point14', 'check point path')  # 체크 포인트 위치 (20190622 16시부터~) log mel, 10만번, multihead 개수 16, layer 개수 12
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point15', 'check point path')  # 체크 포인트 위치 (20190625 16시부터~) mel, 데이터셋 1개만, 5만번
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point16', 'check point path')  # 체크 포인트 위치 (20190625 16.5시부터~) log mel, 데이터셋 1개만, 10만번
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point17', 'check point path')  # 체크 포인트 위치 (20190625 16.5시부터~) log mel, 데이터셋 1개만, 10만번, test뽑아보기
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point18', 'check point path')  # 체크 포인트 위치 (20190625 20시부터~) mel, 데이터셋 1개만, 100000번, logit 지움
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point19', 'check point path')  # 체크 포인트 위치 (20190625 20시부터~) log mel, 데이터셋 1개만, 100000번, logit 지움
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point20', 'check point path')  # 체크 포인트 위치 (20190625 20시부터~) log s^2 mel, 데이터셋 1개만, 100000번, logit 지움
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point21', 'check point path')  # 체크 포인트 위치 (20190625 20시부터~) log mel, 데이터셋 1개만, 100000번, logit 지움, head16, layer12
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point22', 'check point path')  # 체크 포인트 위치 (20190625 20시부터~) log s^2 mel, 데이터셋 1개만, 100000번, logit 지움, head16, layer12
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point23', 'check point path')  # 체크 포인트 위치 (20190625 20시부터~) mel, 데이터셋 1개만, 100000번, logit 지움, head16, layer12
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point24', 'check point path')  # 체크 포인트 위치 (20190627 10시부터~) mel, 데이터셋 1개만, 100000번, head4, layer6
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point25', 'check point path')  # 체크 포인트 위치 (20190627 10시부터~) log mel, 데이터셋 1개만, 100000번, head4, layer6
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point26', 'check point path')  # 체크 포인트 위치 (20190627 10시부터~) log s^2 mel, 데이터셋 1개만, 100000번, head4, layer6
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point27', 'check point path')  # 체크 포인트 위치 (20190627 10.5시부터~) mel, 데이터셋 1개만, 100000번, head4, layer3
# tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point28', 'check point path')  # 체크 포인트 위치 (20190627 10.5시부터~) log mel, 데이터셋 1개만, 100000번, head4, layer3
tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point29', 'check point path')  # 체크 포인트 위치 (20190627 10.5시부터~) log s^2 mel, 데이터셋 1개만, 100000번, head4, layer3

# Define FLAGS
DEFINES = tf.app.flags.FLAGS