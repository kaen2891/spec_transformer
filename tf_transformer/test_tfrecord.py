import tensorflow as tf
import numpy as np
import glob
import matplotlib
import librosa
import librosa.display
tf.enable_eager_execution()
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hop = 256

def rearrange(input, output, target):
    features = {"input": input, "output": output}
    return features, target

def plot_wav(spectrogram, name, save_dir):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), y_axis='hz', x_axis='time', sr=16000,
                             hop_length=hop)
    plt.title(name)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_dir + name + '.png')

def return_2d(time, fft, raw):
    bytes_to_float = np.frombuffer(raw, dtype=np.float32)  # 111872, 1 data
    construct_2d = bytes_to_float.reshape([int(time), int(fft)])
    return construct_2d

NUM_EPOCHS = 5000
filename_inp_enc = './train_input_enc_test2.tfrecords'
filename_oup_dec = './train_output_dec_test2.tfrecords'
filename_trg_dec = './train_target_dec_test2.tfrecords'

def parser(serialized_example):
    """"Parses a single tf.Example into image and label tensors."""
    features = {
        # 'train_input_enc/time': tf.FixedLenFeature((), tf.float32),
        # 'train_input_enc/fft': tf.FixedLenFeature((), tf.float32),
        'train_input_enc/data_raw': tf.FixedLenFeature((), tf.string),
        # 'train_output_dec/time': tf.FixedLenFeature((), tf.float32),
        # 'train_output_dec/fft': tf.FixedLenFeature((), tf.float32),
        'train_output_dec/data_raw': tf.FixedLenFeature((), tf.string),
        # 'train_target_dec/time': tf.FixedLenFeature((), tf.float32),
        # 'train_target_dec/fft': tf.FixedLenFeature((), tf.float32),
        'train_target_dec/data_raw': tf.FixedLenFeature((), tf.string)
    }

    parsed = tf.parse_single_example(serialized_example, features)

    inp_enc_spec = tf.decode_raw(parsed['train_input_enc/data_raw'], tf.float32)
    oup_dec_spec = tf.decode_raw(parsed['train_output_dec/data_raw'], tf.float32)
    trg_dec_spec = tf.decode_raw(parsed['train_target_dec/data_raw'], tf.float32)
    #image = tf.decode_raw(features['image_raw'], tf.uint8)
    #image.set_shape([28 * 28])
    #time = features['train_input_enc/time']
    #fft = features['train_input_enc/fft']
    #spec_shape = tf.stack([time, fft])
    #inp_enc_spec2 = tf.reshape(inp_enc_spec, spec_shape)
    inp_enc_spectrogram = tf.reshape(inp_enc_spec, [437, 256])
    oup_dec_spectrogram = tf.reshape(oup_dec_spec, [437, 256])
    trg_dec_spectrogram = tf.reshape(trg_dec_spec, [437, 256])
    print(
        "inp size = {}, oup size = {}, trg size = {}".format(np.shape(inp_enc_spectrogram), np.shape(oup_dec_spectrogram),
                                                             np.shape(trg_dec_spectrogram)))

    # Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
    # image = tf.cast(image, tf.float32) / 255 - 0.5
    # label = tf.cast(features['label'], tf.int32)
    return inp_enc_spectrogram, oup_dec_spectrogram, trg_dec_spectrogram

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # filename_queue = tf.train.string_input_producer([filename_inp_enc, filename_oup_dec, filename_trg_dec],
    #                                             num_epochs=NUM_EPOCHS)
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)
    #
    # feature = {'train_input_enc/time': tf.FixedLenFeature([], tf.float32),
    #            'train_input_enc/fft': tf.FixedLenFeature([], tf.float32),
    #            'train_input_enc/data_raw': tf.FixedLenFeature([], tf.string),
    #            'train_output_dec/time': tf.FixedLenFeature([], tf.float32),
    #            'train_output_dec/fft': tf.FixedLenFeature([], tf.float32),
    #            'train_output_dec/data_raw': tf.FixedLenFeature([], tf.string),
    #            'train_target_dec/time': tf.FixedLenFeature([], tf.float32),
    #            'train_target_dec/fft': tf.FixedLenFeature([], tf.float32),
    #            'train_target_dec/data_raw': tf.FixedLenFeature([], tf.string)
    #            }
    batch_size = 30
    dataset = tf.data.TFRecordDataset([filename_inp_enc, filename_trg_dec, filename_oup_dec])
    dataset = dataset.shuffle(900)
    dataset = dataset.map(map_func=parser)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    #dataset = dataset.shuffle(900)
    #dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.map(rearrange)
    #print(sess.run(dataset))
    print('done')
    iterator = dataset.make_initializable_iterator()
    next_elem = iterator.get_next()
    #print("x,y=",x,y,)

    sess.run(iterator.initializer)
    print(sess.run(next_elem))
    #z_data = sess.run(z)

    #print(x_data.shape, x_data.dtype)



    print('done')
    print('--')
    #dataset = dataset.repeat(NUM_EPOCHS)

    #dataset = dataset.map(parser().batch(batch_size))
    #dataset = dataset.map(parser, num_threads=1, output_buffer_size=batch_size)


    #features = tf.parse_single_example(serialized_example, features=feature)

    # inp_enc_spec = tf.decode_raw(features['train_input_enc/data_raw'], tf.float32)
    # oup_dec_spec = tf.decode_raw(features['train_output_dec/data_raw'], tf.float32)
    # trg_dec_spec = tf.decode_raw(features['train_target_dec/data_raw'], tf.float32)

    # inp_enc_spectrogram = tf.reshape(inp_enc_spec, [437, 256])
    # oup_dec_spectrogram = tf.reshape(oup_dec_spec, [437, 256])
    # trg_dec_spectrogram = tf.reshape(trg_dec_spec, [437, 256])
    # print("inp size = {}, oup size = {}, trg size = {}".format(np.shape(inp_enc_spectrogram), np.shape(oup_dec_spectrogram),
    #                                                            np.shape(trg_dec_spectrogram)))
    #
    # batch_size = 30

    # inp_enc, oup_dec, trg_dec = tf.train.shuffle_batch([inp_enc_spectrogram, oup_dec_spectrogram, trg_dec_spectrogram],
    #                                                    batch_size=batch_size, capacity=100, num_threads=5, min_after_dequeue=20)

    dataset = tf.data.Dataset.from_tensor_slices((inp_enc_spectrogram, oup_dec_spectrogram, trg_dec_spectrogram))
    print("after slices", dataset)
    # 전체 데이터를 썩는다.

    dataset = dataset.shuffle(buffer_size=900)
    print("after shuffle",dataset)
    # 배치 인자 값이 없다면  에러를 발생 시킨다.
    assert batch_size is not None, "train batchSize must not be None"
    # from_tensor_slices를 통해 나눈것을배치크기 만큼 묶어 준다.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    print("after batch",dataset)
    # 데이터 각 요소에 대해서 rearrange 함수를통해서 요소를 변환하여 맵으로 구성한다.
    dataset = dataset.map(rearrange)
    print("after rearrange",dataset)
    # repeat()함수에 원하는 에포크 수를 넣을수 있으면아무 인자도 없다면 무한으로 이터레이터 된다.
    dataset = dataset.repeat()
    print("after repeat",dataset)
    # make_one_shot_iterator를 통해 이터레이터를 만들어 준다.
    iterator = dataset.make_initializable_iterator()
    print("iterator",iterator)
    # iterator = dataset.make_one_shot_iterator()

    print('done')
    print('----')
    #
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)
    #
    # for batch_index in range(5):
    #     inp_enc, oup_dec, trg_dec = sess.run([inp_enc, oup_dec, trg_dec])
    #     print("input_enc {}, output_dec {} target_dec {}".format(inp_enc, oup_dec, trg_dec))
    #     # img = img.astype(np.float32)
    #     plot_wav(inp_enc, str(batch_index) + 'inp_enc', './' + str(batch_index) + 'inp_enc')
# spectrogram = tf.reshape(spectrogram,[437,256])


#