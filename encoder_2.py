import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, CuDNNLSTM, Input, Embedding, TimeDistributed, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf

# vec_len       = 300   # Length of the vector that we willl get from the embedding layer
# # latent_dim    = 1024  # Hidden layers dimension
# # dropout_rate  = 0.2   # Rate of the dropout layers
# # batch_size    = 64    # Batch size
# # epochs        = 30    # Number of epochs
# #
# # encoder_input = Input(shape=(None,))
# #
# # # Hidden layers of the encoder :
# # encoder_embedding = Embedding(input_dim=10, output_dim=vec_len)(encoder_input)
# # encoder_dropout = (TimeDistributed(Dropout(rate = dropout_rate)))(encoder_embedding)
# # encoder_LSTM = CuDNNLSTM(latent_dim, return_sequences=True)(encoder_dropout)
# #
# # # Output layer of the encoder :
# # encoder_LSTM2_layer = CuDNNLSTM(latent_dim, return_state=True)
# # encoder_outputs, state_h, state_c = encoder_LSTM2_layer(encoder_LSTM)
# #
# # # We discard `encoder_outputs` and only keep the states.
# # encoder_states = [state_h, state_c]


model = Sequential()
model.add(Embedding(1000, 512, input_length=40))
# model.add(Embedding(input_dim=))
input_array = np.random.randint(1000, size=(32, 40))
print("input_array shpae {}".format(np.shape(input_array)))
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
print("output shape {}".format(np.shape(output_array)))
assert output_array.shape == (32, 40, 512)
print("output shape {}".format(np.shape(output_array)))
