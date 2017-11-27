'''
@author: wanzeyu

@contact: wan.zeyu@outlook.com

@file: lstm_test.py

@time: 2017/11/22 12:50
'''
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
import numpy as np
import matplotlib.pyplot as plt

from lstm_vae import create_lstm_vae
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences


def get_data():
    data = np.fromfile('sample_data.dat').reshape(419, 13)
    print(np.shape(data))
    timesteps = 3
    dataX = []
    for i in range(len(data) - timesteps - 1):
        x = data[i:(i + timesteps), :]
        dataX.append(x)
    return np.array(dataX)


x = get_data()
input_dim = x.shape[-1]  # 13
timesteps = x.shape[1]  # 3
batch_size = 10
x = Input(shape=(None, timesteps, input_dim))
h = LSTM(32)(x)
o = Dense(timesteps, activation="sigmoid")(h)
lstm_test = Model(x, o)
optimizer = RMSprop(lr=0.01)
lstm_test.compile(optimizer=optimizer, loss="categorical_crossentropy")
# lstm_test.fit(x, batch_size=batch_size, epochs=1)
lstm_test.predict(x[3], batch_size=batch_size)
