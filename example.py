import numpy as np
from lstm_vae import create_lstm_vae
from lstm_vae.helper import get_data, print_sentence_with_w2v, padding_len

np.set_printoptions(threshold=np.nan)

max_time_step = 100

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
                                    intermediate_dim=20,
                                    latent_dim=20,
                                    epsilon_std=1.,
                                    max_time_step=max_time_step)

    # plot_model(enc, to_file="encoder.png")
    # plot_model(vae, to_file="vae.png")
    # plot_model(gen, to_file="generator.png")

    # print("start loading weights")
    # vae.load_weights("vae_model.h5")
    # enc.load_weights("enc.h5")
    # gen.load_weights("gen.h5")
    # print("loading weights finished")

    for i in range(200):
        vae.fit([original, paraphrase], original, epochs=1, batch_size=batch_size)

        vae.save_weights("vae_model.h5")
        enc.save_weights("enc.h5")
        gen.save_weights("gen.h5")

        print("encoding start")
        sent_encoded = enc.predict([original, paraphrase], batch_size=batch_size)
        print(sent_encoded.shape)
        print("encoding finished")

        print("decoding start")
        sent_decoded = gen.predict([sent_encoded, original], batch_size=batch_size)
        print("decoding finished")
        print(sent_decoded.shape)
        print_sentence_with_w2v(sent_decoded)

    # test_hom = shortest_homology(sent_encoded[3], sent_encoded[10], 5)
    # for point in test_hom:
    #     p = gen.predict(np.array([point]))[0]
    #     print_sentence_with_w2v(p)
    #
    # test_hom = shortest_homology(sent_encoded[2], sent_encoded[1500], 20)
    # for point in test_hom:
    #     p = gen.predict(np.array([find_similar_encoding(point)]))[0]
    #     print_sentence_with_w2v(p)
