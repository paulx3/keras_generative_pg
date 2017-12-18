import numpy as np

from lstm_vae import create_lstm_vae
from lstm_vae.helper import get_data_v2, padding_len, vocab, manual_one_hot, id_vocab
from numpy import array
from keras.preprocessing.sequence import pad_sequences

np.set_printoptions(threshold=np.nan)

if __name__ == "__main__":
    original = pad_sequences(get_data_v2("test_source.txt"), padding="post", value=0.0, maxlen=30)
    paraphrase = pad_sequences(get_data_v2("test_target.txt"), padding="post", value=0.0, maxlen=30)
    target_paraphrase = manual_one_hot(paraphrase)
    input_dim = len(vocab)
    # time_steps = original.shape[1]
    time_steps = padding_len + 1
    batch_size = 1
    print("time_steps:" + str(time_steps))
    print("input_dim:" + str(input_dim))
    vae, enc, gen = create_lstm_vae(input_dim,
                                    timesteps=time_steps,
                                    batch_size=batch_size,
                                    intermediate_dim=800,
                                    latent_dim=800,
                                    epsilon_std=1.,
                                    )

    # plot_model(enc, to_file="encoder.png", show_shapes=True)
    # plot_model(vae, to_file="vae.png", show_shapes=True)
    # plot_model(gen, to_file="generator.png", show_shapes=True)

    # print("start loading weights")
    # vae.load_weights("vae_model.h5")
    # enc.load_weights("enc.h5")
    # gen.load_weights("gen.h5")
    # print("loading weights finished")
    for i in range(200):
        vae.fit([original, paraphrase], target_paraphrase, epochs=1, batch_size=batch_size, shuffle=True,
                validation_split=0.2
                # validation_data=([test_original, test_paraphrase], test_paraphrase),
                # callbacks=[TensorBoard(log_dir="./logs")]
                )

        # vae.save_weights("vae_model.h5")
        # enc.save_weights("enc.h5")
        # gen.save_weights("gen.h5")

        for t in range(10):
            # print("encoding start")
            sent_encoded = enc.predict([original[t:t + 1], paraphrase[t:t + 1]], batch_size=batch_size)
            print(sent_encoded.shape)
            # print("encoding finished")

            # print("decoding start")
            sent_decoded = gen.predict([sent_encoded, original[t:t + 1]], batch_size=batch_size)
            for n in sent_decoded[0]:
                print(id_vocab[np.argmax(n)], end=" ")
            # print("decoding finished")
            print()
