# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
from layer import FFN
from attention import Attention
from configs import DEFINES
import numpy as np

def positional_encoding(dim, sentence_length):
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim)
                            for pos in range(sentence_length) for i in range(dim)])

    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=tf.float32)


class Encoder(tf.keras.Model):
    def __init__(self,
                 num_layers=6,
                 num_heads=8,
                 linear_key_dim=16,
                 linear_value_dim=16,
                 model_dim=128,
                 ffn_dim=512,
                 dropout=0.2):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout

    def build(self, encoder_inputs):
        o1 = tf.identity(encoder_inputs)

        for i in range(1, self.num_layers + 1):
            with tf.variable_scope("layer"):
                o2 = self._add_and_norm(o1, self._self_attention(q=o1,
                                                                 k=o1,
                                                                 v=o1), num=1)
                o3 = self._add_and_norm(o2, self._positional_feed_forward(o2), num=2)
                o1 = tf.identity(o3)

        return o3

    def _self_attention(self, q, k, v):
        with tf.variable_scope("self-attention"):
            attention = Attention(num_heads=self.num_heads,
                                  masked=False,
                                  linear_key_dim=self.linear_key_dim,
                                  linear_value_dim=self.linear_value_dim,
                                  model_dim=self.model_dim,
                                  dropout=self.dropout)
            return attention.multi_head(q, k, v)

    def _add_and_norm(self, x, sub_layer_x, num=0):
        with tf.variable_scope("add-and-norm"):
            return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x))  # with Residual connection


class Decoder:
    """Decoder class"""

    def __init__(self,
                 num_layers=6,
                 num_heads=8,
                 linear_key_dim=16,
                 linear_value_dim=16,
                 model_dim=128,
                 ffn_dim=512,
                 dropout=0.2):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout

    def build(self, decoder_inputs, encoder_outputs):
        o1 = tf.identity(decoder_inputs)

        for i in range(1, self.num_layers+1):
            with tf.variable_scope("layer"):
                o2 = self._add_and_norm(o1, self._masked_self_attention(q=o1,
                                                                        k=o1,
                                                                        v=o1), num=1)
                o3 = self._add_and_norm(o2, self._encoder_decoder_attention(q=o2,
                                                                            k=encoder_outputs,
                                                                            v=encoder_outputs), num=2)
                o4 = self._add_and_norm(o3, self._positional_feed_forward(o3), num=3)
                o1 = tf.identity(o4)

        return o4

    def _masked_self_attention(self, q, k, v):
        with tf.variable_scope("masked-self-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    masked=True,  # Not implemented yet
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout)
            return attention.multi_head(q, k, v)

    def _add_and_norm(self, x, sub_layer_x, num=0):
        with tf.variable_scope("add-and-norm"):
            return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x)) # with Residual connection

    def _encoder_decoder_attention(self, q, k, v):
        with tf.variable_scope("encoder-decoder-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    masked=False,
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout)
            return attention.multi_head(q, k, v)

    def _positional_feed_forward(self, output):
        with tf.variable_scope("feed-forward"):
            ffn = FFN(w1_dim=self.ffn_dim,
                      w2_dim=self.model_dim,
                      dropout=self.dropout)
            return ffn.dense_relu_dense(output)

def Model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    # num_layers = 6,
    # num_heads = 8,
    # linear_key_dim = params['embedding_size'] / params['attention_head_size']
    linear_key_dim = params['embedding_size'] / params['attention_head_size']
    # model_dim = 128,
    # ffn_dim = 512,
    # dropout = 0.2
    position_encode = positional_encoding(params['embedding_size'], params['max_sequence_length'])

    # embedding = tf.keras.layers.Embedding(params['vocabulary_length'], params['embedding_size'])

    encoder_layers = Encoder(params['layer_size'],params['attention_head_size'],linear_key_dim, linear_key_dim, params['model_hidden_size'], params['ffn_hidden_size'],
                      params['dropout_width'])

    decoder_layers = Decoder(params['layer_size'],params['attention_head_size'],linear_key_dim, linear_key_dim, params['model_hidden_size'], params['ffn_hidden_size'],
                      params['dropout_width'])

    # logit_layer = tf.keras.layers.Dense(params['vocabulary_length'])

    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        # x_embedded_matrix = embedding(features['input']) + position_encode
        x_embedded_matrix = features['input'] + position_encode
        encoder_outputs = encoder_layers(x_embedded_matrix)

    loop_count = params['max_sequence_length'] if PREDICT else 1

    output, logits = None, None

    for i in range(loop_count):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            if i > 0:
                output = tf.ones((output.shape[0], 1), dtype=tf.float32)
            else:
                output = features['output']

            # y_embedded_matrix = embedding(output) + position_encode
            y_embedded_matrix = output + position_encode
            decoder_outputs = decoder_layers(y_embedded_matrix, encoder_outputs)

            # logits = decoder_layers(decoder_outputs)
            # mel_loss = tf.reduce_mean(tf.abs(decod - mel_targets))
            logits = decoder_outputs
            # predict = tf.argmax(logits, 2)
            # pred = tf.add(tf.multiply())

    if PREDICT:
        predictions = {
            # 'indexs': predict,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = tf.reduce_mean(tf.abs(decoder_outputs - labels))
    # loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
    # loss = tf.reduce_mean(tf.nn.m(logits=logits, labels=labels))
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=logits, labels=tf.argmax(tf.cast(labels, dtype=tf.int32), 1)))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    #     logits=logits, labels=labels))
    # accuracy = tf.metrics.accuracy(labels=labels, predictions=predict, name='accOp')

    # metrics = {'accuracy': accuracy}
    # tf.summary.scalar('accuracy', accuracy[1])

    if EVAL:
        # return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    assert TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)