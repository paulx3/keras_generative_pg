import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing.sequence import pad_sequences
from keras import objectives
import numpy as np


def create_lstm_vae(input_dim,
                    timesteps,
                    batch_size,
                    intermediate_dim,
                    latent_dim,
                    epsilon_std=1.,
                    max_time_step=100, ):
    """
    Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator. 

    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM. 
        latent_dim: int, latent z-layer shape. 
        epsilon_std: float, z-layer sigma.


    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """

    def embedding_lstm(dimension, max_len):
        def func(x):
            if dimension == 0:
                pass
            if dimension == 1:
                pass
            if dimension == 2:
                if x[2] is not None:
                    print(x.shape)
                    # if x[2].shape[0] != max_len:
                    #     cur_len = x[2].shape[0]
                    #     zeroes = max_len - cur_len
            if dimension == 3:
                pass
            if dimension == 4:
                pass

        return Lambda(func)

    def print_shape():
        def func(x):
            if x is not None:
                print(x.shape)

        return Lambda(func)

    def crop(dimension, start, end):
        # Crops (or slices) a Tensor on a given dimension from start to end
        # example : to crop tensor x[:, :, 5:10]
        # call slice(2, 5, 10) as you want to crop on the second dimension
        def func(x):
            if dimension == 0:
                return x[start: end]
            if dimension == 1:
                return x[:, start: end]
            if dimension == 2:
                return x[:, :, start: end]
            if dimension == 3:
                return x[:, :, :, start: end]
            if dimension == 4:
                return x[:, :, :, :, start: end]

        return Lambda(func)

    x = Input(shape=(timesteps, input_dim,), name="EncoderInput_1")

    # original sentence encoding
    original_sentence_input = Input(shape=(timesteps, input_dim,), name="OriginalSentenceInput_1")
    layer_1 = LSTM(intermediate_dim)(original_sentence_input)

    # LSTM encoding
    h = LSTM(intermediate_dim)(x)
    # h = embedding_lstm(dimension=2, max_len=max_time_step)(x)
    # h = LSTM(max_time_step)(x)
    # merge original and paraphrase
    h = keras.layers.concatenate([layer_1, h], axis=-1)
    # h = crop(2, 0, max_time_step)(h)
    # VAE Z layer
    # h = Dense(intermediate_dim)(h)
    # h = LSTM(intermediate_dim)(h)
    h = RepeatVector(timesteps)(h)
    h = LSTM(intermediate_dim)(h)
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_sigma) * epsilon_std

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    # decoded LSTM layer
    decoder_h = LSTM(timesteps, return_sequences=True, name="DecoderH_1")
    decoder_mean = LSTM(input_dim, return_sequences=True, name="DecoderMean_1")

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = Model([x, original_sentence_input], x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model([x, original_sentence_input], z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,), name="DecoderLatentInput_1")
    decoder_input_repeated = RepeatVector(timesteps)(decoder_input)
    # decoder original encoder
    decoder_original_input = Input(shape=(timesteps, input_dim,), name="DecoderOriginalInput_1")

    decoder_layer1 = LSTM(timesteps)(decoder_original_input)

    _h_decoded = RepeatVector(timesteps)(decoder_layer1)
    _h_decoded = decoder_h(_h_decoded)
    _x_decoded_mean = decoder_mean(_h_decoded)
    output = keras.layers.concatenate([_x_decoded_mean, decoder_input_repeated], axis=-1)
    output = Dense(timesteps)(output)
    # output = crop(2, 0, 30)(output)
    output = decoder_h(output)
    output = decoder_mean(output)
    generator = Model([decoder_input, decoder_original_input], output)

    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    vae.compile(optimizer='rmsprop', loss=vae_loss)

    return vae, encoder, generator
