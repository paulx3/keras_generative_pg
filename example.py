import numpy as np
from lstm_vae import create_lstm_vae
from lstm_vae.helper import get_data, print_sentence_with_w2v, padding_len
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard

np.set_printoptions(threshold=np.nan)

if __name__ == "__main__":
    original = get_data("test_source.txt")
    paraphrase = get_data("test_target.txt")
    input_dim = original.shape[-1]
    # time_steps = original.shape[1]
    time_steps = padding_len
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
    train_original = original[:800]
    train_paraphrase = paraphrase[:800]
    test_original = original[800:]
    test_paraphrase = paraphrase[800:]
    for i in range(200):
        vae.fit([train_original, train_paraphrase], train_paraphrase, epochs=1, batch_size=batch_size, shuffle=True,
                # validation_data=([test_original, test_paraphrase], test_paraphrase),
                # callbacks=[TensorBoard(log_dir="./logs")]
                )

        vae.save_weights("vae_model.h5")
        enc.save_weights("enc.h5")
        gen.save_weights("gen.h5")

        print("encoding start")
        sent_encoded = enc.predict([test_original, test_paraphrase], batch_size=batch_size)
        print(sent_encoded.shape)
        print("encoding finished")

        print("decoding start")
        sent_decoded = gen.predict([sent_encoded, test_original], batch_size=batch_size)
        print("decoding finished")
        print(sent_decoded.shape)
        print_sentence_with_w2v(sent_decoded)