# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras_multi_head import MultiHeadAttention
from keras_layer_normalization import LayerNormalization
from keras_position_wise_feed_forward import FeedForward
from keras.layers import Input, Dense, Dropout, Masking
import make_transformer
from keras.models import *
tf.reset_default_graph()

MODEL_SAVE_FOLDER_PATH = './transformer/model/'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

ACC_SAVE_FOLDER_PATH = './transformer/loss/'
if not os.path.exists(ACC_SAVE_FOLDER_PATH):
  os.mkdir(ACC_SAVE_FOLDER_PATH)

parser = argparse.ArgumentParser(description='Inference script')
parser.add_argument("--epoch", type=int, default=2,
                    help="train epoch")
parser.add_argument("--batch_size", type=int, default=3,
                    help="train batch size")
parser.add_argument("--drop_out", type=int, default=0.2, help="drop rate")
parser.add_argument("--learning_rate", type=int, default=1e-5, help="train learning rate")
parser.add_argument("--early_patience", type=int, default=30, help="train learning patience")

new_args = vars(parser.parse_args())

os.environ['CUDA_VISIBLE_DEVICES']='7'

import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback
from keras.models import Sequential
from keras.layers import ReLU, LeakyReLU, Dropout

def load_data():
    print()
    with open('/mnt/junewoo/workspace/transform/pair_1_dataset/train/trainmale', 'rb') as file:
        X_train = pickle.load(file)
    with open('/mnt/junewoo/workspace/transform/pair_1_dataset/train/trainfemale', 'rb') as file:
        Y_train = pickle.load(file)
    # X_train = None
    # Y_train = None
    with open('/mnt/junewoo/workspace/transform/pair_1_dataset/valid/validmale', 'rb') as file:
        X_val = pickle.load(file)
    with open('/mnt/junewoo/workspace/transform/pair_1_dataset/valid/validfemale', 'rb') as file:
        Y_val = pickle.load(file)
    # X_val = None
    # Y_val = None
    with open('/mnt/junewoo/workspace/transform/pair_1_dataset/test/testmale', 'rb') as file:
        X_test = pickle.load(file)
    with open('/mnt/junewoo/workspace/transform/pair_1_dataset/test/testfemale', 'rb') as file:
        Y_test = pickle.load(file)
    # X_test = None
    # Y_test = None
    X_train_T = []
    Y_train_T = []
    for i in range(len(X_train)):
        X_train[i] = X_train[i].T
        Y_train[i] = Y_train[i].T
        X_train_T.append(X_train[i])
        Y_train_T.append(Y_train[i])

    X_val_T = []
    Y_val_Y = []
    for i in range(len(X_val)):
        X_val[i] = X_val[i].T
        Y_val[i] = Y_val[i].T
        X_val_T.append(X_val[i])
        Y_val_Y.append(Y_val[i])

    X_test_T = []
    Y_test_T = []
    for i in range(len(X_test)):
        X_test[i] = X_test[i].T
        Y_test[i] = Y_test[i].T
        X_test_T.append(X_test[i])
        Y_test_T.append(Y_test[i])
    return X_train_T, Y_train_T, X_val_T, Y_val_Y, X_test_T, Y_test_T

def model(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, n_head, n_encoder, n_decoder):

    input_dim = 128 # mel_bin
    time_step = 60
    # output_dim = 128 # mel_bin
    # feature_dim = 60
    # input_shape = (None, input_dim)
    # inp = Input(shape=(feature_dim,input_dim))
    # # x = Embedding(config["max_features"], config["embed_size"], weights=[embedding_matrix],
    # #               trainable=config["trainable"])(inp)
    # multi_head_attn_layer = make_transformer.Attention(8, 16)(
    #     [inp, inp, inp])  # output: [batch_size, time_step, nb_head*size_per_head]
    # # model = Sequential()
    # x = Dropout(0.1)(multi_head_attn_layer)
    # x = Dense(1, activation='sigmoid')(x)
    # model = Model(inputs=inp, outputs=x)

    #X_train: (7, 60, 128)
    # masked_inputs = Masking(0.0)
    model = Sequential()
    # model.add(Masking(mask_value=0., input_shape=(time_step, input_dim)))
    # input_layer = keras.layers.Input(Masking(mask_value=0., input_shape=(feature_dim, input_dim)))
    #
    input_layer = keras.layers.Input(
        shape=(time_step, input_dim), #60, 128
        name='Input',
    )
    for i in range(n_head):
        
    # model.add(MultiHeadAttention(
    #     head_num=8,
    #     name='Multi-Head',
    # ))
    att_layer = MultiHeadAttention(
        head_num=8,
        activation=None,
        history_only=False,
        trainable=True,
        name='Multi-Head',
    )(input_layer)
    # model.add(Dropout(0.1))
    drop_out = Dropout(0.1)(att_layer)
    # ADD & Norm
    Norm_1 = LayerNormalization(trainable=True)(drop_out)
    # Feed Forward
    FFN_1 = FeedForward(units=128, activation='relu', trainable=True, name='Feed-Forward')(Norm_1)
    # ADD & Norm
    Norm_2 = LayerNormalization(trainable=True)(FFN_1)


    model = keras.models.Model(inputs=input_layer, outputs=Norm_2)
    model.compile(
        loss="binary_crossentropy",
        # optimizer = Adam(lr = config["lr"], decay = config["lr_d"]),
        optimizer='adam',
        metrics=["accuracy"])

    # training
    # decay_rate = new_args['learning_rate'] / new_args['num_epochs']

    # adam = optimizers.Adam(lr=new_args['learning_rate'], beta_1=0.9, beta_2=0.98, epsilon=1e-9, decay=decay_rate)

    # model.compile(loss='keras.losses.sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    early_stopping = EarlyStopping(patience=new_args['early_patience'])
    print(model.summary())
    plot_model(model, to_file='./model_transformer.png')
    hist = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=new_args['epoch'],
                     batch_size=new_args['batch_size'], shuffle=True, callbacks=[early_stopping])
    # hist = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),epochs=new_args['epoch'], batch_size=argparse['batch_size'],shuffle=True, callbacks=[early_stopping])

    print('\nAccuracy: {:.4f}'.format(model.evaluate(X_valid, Y_valid)[1]))

    scores = model.evaluate(X_test, Y_test)
    print("-- Evaluate --")
    print("test {}: {:.2f}".format(model.metrics_names[1], scores[1] * 100))

    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.savefig(
        ACC_SAVE_FOLDER_PATH + 'transformer_acc:{:.4f}_test:_{:.2f}.png'.format(model.evaluate(X_valid,Y_valid)[1],scores[1] * 100))
    plt.show()


def main():
    # X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data()
    # for i in range(len(X_train)):
    #     print("{} 번째 길이: {}".format(i, len(X_train[i])))
    #
    # for j in range(len(X_val)):
    #     print("{} 번째 길이: {}".format(j, len(X_val[j])))
    #
    # for k in range(len(X_test)):
    #     print("{} 번째 길이: {}".format(k, len(X_test[k])))

    # # 60,128 dim select
    # new_X_train = []
    # new_Y_train = []
    # new_X_train.append(X_train[2])
    # new_X_train.append(X_train[3])
    # new_X_train.append(X_train[7])
    # new_X_train.append(X_train[9])
    # new_X_train.append(X_train[10])
    # new_X_train.append(X_train[14])
    # new_X_train.append(X_train[20])
    #
    # new_Y_train.append(Y_train[2])
    # new_Y_train.append(Y_train[3])
    # new_Y_train.append(Y_train[7])
    # new_Y_train.append(Y_train[9])
    # new_Y_train.append(Y_train[10])
    # new_Y_train.append(Y_train[14])
    # new_Y_train.append(Y_train[20])
    #
    # new_X_val = []
    # new_Y_val = []
    #
    # new_X_val.append(X_val[0])
    # new_Y_val.append(Y_val[0])
    #
    # new_X_test = []
    # new_Y_test = []
    #
    # new_X_test.append(X_test[0])
    # new_X_test.append(X_test[1])
    # new_X_test.append(X_test[2])
    # new_Y_test.append(Y_test[0])
    # new_Y_test.append(Y_test[1])
    # new_Y_test.append(Y_test[2])
    #
    # with open('./test_head/x_train', 'wb') as fp:
    #     pickle.dump(new_X_train, fp)
    # with open('./test_head/y_train', 'wb') as fp:
    #     pickle.dump(new_Y_train, fp)
    # with open('./test_head/x_val', 'wb') as fp:
    #     pickle.dump(new_X_val, fp)
    # with open('./test_head/y_val', 'wb') as fp:
    #     pickle.dump(new_Y_val, fp)
    # with open('./test_head/x_test', 'wb') as fp:
    #     pickle.dump(new_X_test, fp)
    # with open('./test_head/y_test', 'wb') as fp:
    #     pickle.dump(new_Y_test, fp)

    # new (60,128)
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
    print("X_train 길이:",len(X_train))

    new_X_train = np.array((X_train))
    new_Y_train = np.array((Y_train))
    new_X_val = np.array((X_val))
    new_Y_val = np.array((Y_val))
    new_X_test = np.array((X_test))
    new_y_test = np.array((Y_test))
    # print(len(new_X_train))
    # print(type(new_X_train))
    # print(type(new_X_train[0]))
    # model(X_train, Y_train, X_val, Y_val, X_test, Y_test)
    model(new_X_train, new_Y_train, new_X_val, new_Y_val, new_X_test, new_y_test)

    # model(X_train[0], Y_train[0], X_val[0], Y_val[0], X_test[0], Y_test[0])
    # print(len(new_X_train))
    # for i in range(len(new_X_train)):
    #     print("{}번째 shape {}".format(i, np.shape(new_X_train[i])))
    '''
    print(np.shape(X_train[0]))
    print(np.shape(X_train[1]))
    # new = np.vstack((X_train[0], X_train[1]))
    sum = X_train[0]
    us = len(X_train[0])
    print(len(X_train))
    for i in range(len(X_train)-1):
        i += 1
        print(i)
        sum = np.vstack((sum, X_train[i]))
        us += len(X_train[i])
        # all_label = np.concatenate((X_train[i]), axis=1)
        # print(np.shape(all_label))
    # print(np.shape(all_label))
    # print(sum)
    print(np.shape(sum))
    print(us)
    length = len(sum)
    mok, na = divmod(length, 50)
    print(mok, na)
    sum2 = sum.reshape(-1, length, 128)
    ########## reshape 하기.
    # print(np.shape(sum2))
    # sum3 = sum.reshape(31, 50, 128)
    # print(np.shape(sum3))
    '''



if __name__ == '__main__':
    main()

