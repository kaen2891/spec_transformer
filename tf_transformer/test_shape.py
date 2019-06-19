import tensorflow as tf
import pickle
import numpy as np

def rearrange(input, output, target):
    features = {"input": input, "output": output}
    return features, target

def train_input_fn(train_input_enc, train_output_dec, train_target_dec, batch_size):
    # print("len_input_enc {} len_output_dec {} len_target_dec {} len_batch_size {}".format(len(train_input_enc), len(train_output_dec), len(train_target_dec), len(batch_size)))
    # Dataset을 생성하는 부분으로써 from_tensor_slices부분은
    # 각각 한 문장으로 자른다고 보면 된다.
    # train_input_enc, train_output_dec, train_target_dec
    # 3개를 각각 한문장으로 나눈다.
    dataset = tf.data.Dataset.from_tensor_slices((train_input_enc, train_output_dec, train_target_dec))
    print("len(train_input_enc)", len(train_input_enc))
    print("dataset 1", dataset)
    print("dataset 1 shape", np.shape(dataset))
    print("dataset 1", type(dataset))
    print("왜 출력 안되냐")
    # 전체 데이터를 썩는다.
    dataset = dataset.shuffle(buffer_size=len(train_input_enc))
    print("dataset 2", dataset)
    print("dataset 2 shape", np.shape(dataset))
    print("dataset 2", type(dataset))
    # 배치 인자 값이 없다면  에러를 발생 시킨다.
    assert batch_size is not None, "train batchSize must not be None"
    # from_tensor_slices를 통해 나눈것을
    # 배치크기 만큼 묶어 준다.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # 데이터 각 요소에 대해서 rearrange 함수를
    # 통해서 요소를 변환하여 맵으로 구성한다.
    dataset = dataset.map(rearrange)
    # repeat()함수에 원하는 에포크 수를 넣을수 있으면
    # 아무 인자도 없다면 무한으로 이터레이터 된다.
    dataset = dataset.repeat()
    # make_one_shot_iterator를 통해 이터레이터를
    # 만들어 준다.
    iterator = dataset.make_one_shot_iterator()
    # 이터레이터를 통해 다음 항목의 텐서
    # 개체를 넘겨준다.
    return iterator.get_next()


# 평가에 들어가 배치 데이터를 만드는 함수이다.
def eval_input_fn(eval_input_enc, eval_output_dec, eval_target_dec, batch_size):
    # Dataset을 생성하는 부분으로써 from_tensor_slices부분은
    # 각각 한 문장으로 자른다고 보면 된다.
    # eval_input_enc, eval_output_dec, eval_target_dec
    # 3개를 각각 한문장으로 나눈다.
    dataset = tf.data.Dataset.from_tensor_slices((eval_input_enc, eval_output_dec, eval_target_dec))

    # 전체 데이터를 섞는다.
    dataset = dataset.shuffle(buffer_size=len(eval_input_enc))
    # 배치 인자 값이 없다면  에러를 발생 시킨다.
    assert batch_size is not None, "eval batchSize must not be None"
    # from_tensor_slices를 통해 나눈것을
    # 배치크기 만큼 묶어 준다.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # 데이터 각 요소에 대해서 rearrange 함수를
    # 통해서 요소를 변환하여 맵으로 구성한다.
    dataset = dataset.map(rearrange)
    # repeat()함수에 원하는 에포크 수를 넣을수 있으면
    # 아무 인자도 없다면 무한으로 이터레이터 된다.
    # 평가이므로 1회만 동작 시킨다.
    dataset = dataset.repeat(1)
    # make_one_shot_iterator를 통해
    # 이터레이터를 만들어 준다.
    iterator = dataset.make_one_shot_iterator()
    # 이터레이터를 통해 다음 항목의
    # 텐서 개체를 넘겨준다.
    return iterator.get_next()


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

# 훈련셋 인코딩 만드는 부분이다.
train_input_enc, train_input_enc_length = np.asarray(X_train), 60

# 훈련셋 디코딩 입력 부분 만드는 부분이다.
train_output_dec, train_output_dec_length = np.asarray(Y_train), 60

# 훈련셋 디코딩 출력 부분 만드는 부분이다.
train_target_dec = np.asarray(Y_train)
# lambda: train_input_fn(train_input_enc, train_output_dec, train_target_dec, DEFINES.batch_size), steps=DEFINES.train_steps)
print(train_input_fn(train_input_enc, train_output_dec, train_target_dec, 1))