import keras
from keras import backend as K
from keras import objectives
from keras.layers import Input, RepeatVector, LSTM, Embedding
from keras.layers.core import Dense, Lambda
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.backend import permute_dimensions, gather


def create_lstm_vae(input_dim,
                    timesteps,
                    batch_size,
                    intermediate_dim,
                    latent_dim,
                    epsilon_std=1., ):
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

    def crop_dimension():
        def func(x):
            x = K.permute_dimensions(x, (1, 0, 2))
            x = K.gather(x, [i for i in range(30)])
            x = K.permute_dimensions(x, (1, 0, 2))
            return x

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

    crop_layer = crop_dimension()
    # drop_out_layer = Dropout(rate=0.0)
    # original sentence encoding
    original_sentence_input = Input(shape=(None,), name="OriginalInput_1")
    # original_sentence_input = Input(shape=(timesteps, input_dim,), name="OriginalInput_1")
    embedding_layer = Embedding(30, input_dim + 1, mask_zero=True)
    original_encoder_layer_1 = LSTM(intermediate_dim, return_sequences=True, name="OriginalEncoderLSTM_1")
    original_encoder_layer_2 = LSTM(intermediate_dim, return_sequences=True, name="OriginalEncoderLSTM_2")
    original_encoder_layer_3 = LSTM(intermediate_dim, return_sequences=True, name="OriginalEncoderLSTM_3")

    encoded_original = embedding_layer(original_sentence_input)
    encoded_original = original_encoder_layer_1(encoded_original)
    # encoded_original = drop_out_layer(encoded_original)
    encoded_original = original_encoder_layer_2(encoded_original)
    # encoded_original = drop_out_layer(encoded_original)
    encoded_original = original_encoder_layer_3(encoded_original)

    # paraphrase sentence encoder
    paraphrase_sentence_encoder_layer1 = LSTM(intermediate_dim, return_sequences=True, name="ParaphraseEncoderLSTM_1")
    paraphrase_sentence_encoder_layer2 = LSTM(intermediate_dim, return_sequences=True, name="ParaphraseEncoderLSTM_2")
    paraphrase_sentence_encoder_layer3 = LSTM(intermediate_dim, return_sequences=True, name="ParaphraseEncoderLSTM_3")

    # x = Input(shape=(timesteps, input_dim,), name="ParaphraseInput_1")
    x = Input(shape=(None,), name="ParaphraseInput_1")
    embedded_x = embedding_layer(x)
    # LSTM encoding
    # h = LSTM(intermediate_dim)(x)
    # merge original and paraphrase
    encoded_original = crop_layer(encoded_original)
    embedded_x = crop_layer(embedded_x)
    h = keras.layers.concatenate([encoded_original, embedded_x], axis=-1)
    h = paraphrase_sentence_encoder_layer1(h)
    # h = drop_out_layer(h)
    h = paraphrase_sentence_encoder_layer2(h)
    # h = drop_out_layer(h)
    h = paraphrase_sentence_encoder_layer3(h)

    # VAE Z layer
    # h = Dense(intermediate_dim)(h)
    # h = LSTM(intermediate_dim, return_sequences=True)(h)
    # h = RepeatVector(timesteps)(h)
    # h = LSTM(intermediate_dim)(h)

    # test begin
    # z_mean = Dense(latent_dim)(h)
    # z_log_sigma = Dense(latent_dim)(h)
    # test ends
    z_mean = LSTM(latent_dim)(h)
    z_log_sigma = LSTM(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        # return z_mean + z_log_sigma * epsilon
        return z_mean + K.exp(z_log_sigma) * epsilon
        # return z_mean + K.exp(z_log_sigma / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    # decoded LSTM layer
    # decoder_h = LSTM(timesteps, return_sequences=True, name="DecoderH_1")
    # decoder_mean = LSTM(input_dim, return_sequences=True, name="DecoderMean_1")
    decoder_h = Dense(intermediate_dim, name="DecoderH_1", activation="relu")
    decoder_mean = Dense(input_dim, name="DecoderMean_1", activation="sigmoid")

    # paraphrase sentence decoder
    paraphrase_sentence_decoder_layer1 = LSTM(intermediate_dim, return_sequences=True,
                                              name="ParaphraseSentenceDecoder_1")
    paraphrase_sentence_decoder_layer2 = LSTM(intermediate_dim, return_sequences=True,
                                              name="ParaphraseSentenceDecoder_2")
    paraphrase_sentence_decoder_layer3 = LSTM(intermediate_dim, return_sequences=True,
                                              name="ParaphraseSentenceDecoder_3")

    h_decoded = RepeatVector(timesteps)(z)

    paraphrase_sentence_decoded = paraphrase_sentence_decoder_layer1(encoded_original)

    Lambda(lambda x: K.permute_dimensions(x, (1, 0, 2)))
    paraphrase_sentence_decoded = keras.layers.concatenate([h_decoded, paraphrase_sentence_decoded],
                                                           axis=-1)
    paraphrase_sentence_decoded = paraphrase_sentence_decoder_layer2(paraphrase_sentence_decoded)
    paraphrase_sentence_decoded = keras.layers.concatenate([h_decoded, paraphrase_sentence_decoded],
                                                           axis=-1)
    paraphrase_sentence_decoded = paraphrase_sentence_decoder_layer3(paraphrase_sentence_decoded)

    h_decoded = decoder_h(paraphrase_sentence_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = Model([original_sentence_input, x], x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model([original_sentence_input, x], z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_latent_input = Input(shape=(latent_dim,), name="DecoderLatentInput_1")
    decoder_input_repeated = RepeatVector(timesteps)(decoder_latent_input)
    # decoder original encoder
    # decoder_original_input = Input(shape=(timesteps, input_dim,), name="DecoderOriginalInput_1")
    decoder_original_input = Input(shape=(None,), name="DecoderOriginalInput_1")
    decoder_original_encoded = embedding_layer(decoder_original_input)
    decoder_original_encoded = original_encoder_layer_1(decoder_original_encoded)
    # decoder_original_encoded = drop_out_layer(decoder_original_encoded)
    decoder_original_encoded = original_encoder_layer_2(decoder_original_encoded)
    # decoder_original_encoded = drop_out_layer(decoder_original_encoded)
    decoder_original_encoded = original_encoder_layer_3(decoder_original_encoded)

    paraphrase_sentence_decoded = paraphrase_sentence_decoder_layer1(decoder_original_encoded)
    paraphrase_sentence_decoded = crop_layer(paraphrase_sentence_decoded)
    paraphrase_sentence_decoded = keras.layers.concatenate([decoder_input_repeated, paraphrase_sentence_decoded],
                                                           axis=-1)
    # paraphrase_sentence_decoded = drop_out_layer(paraphrase_sentence_decoded)
    paraphrase_sentence_decoded = paraphrase_sentence_decoder_layer2(paraphrase_sentence_decoded)
    paraphrase_sentence_decoded = keras.layers.concatenate([decoder_input_repeated, paraphrase_sentence_decoded],
                                                           axis=-1)
    # paraphrase_sentence_decoded = drop_out_layer(paraphrase_sentence_decoded)
    paraphrase_sentence_decoded = paraphrase_sentence_decoder_layer3(paraphrase_sentence_decoded)

    _h_decoded = decoder_h(paraphrase_sentence_decoded)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model([decoder_latent_input, decoder_original_input], _x_decoded_mean)

    # def vae_loss(x, x_decoded_mean):
    #     xent_loss = objectives.mse(x, x_decoded_mean)
    #     kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
    #     loss = xent_loss + kl_loss
    #     return loss
    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    # def vae_loss(x, x_decoded_mean):
    #     xent_loss = input_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    #     kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    #     return K.mean(xent_loss + kl_loss)
    # def vae_loss(x, x_decoded_mean):
    #     # E[log P(X|z)]
    #     recon = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=1)
    #     # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    #     kl = 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma, axis=1)
    #     return recon + kl

    # sgd = SGD(lr=0.00005, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.00005)
    # rmsprop = RMSprop()
    vae.compile(optimizer=sgd, loss=vae_loss)

    return vae, encoder, generator
