import numpy as np
import matplotlib.pyplot as plt
from lstm_vae import create_lstm_vae
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models import KeyedVectors
from keras import backend as K
from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Lambda, Layer, LSTM
from keras.models import Model
from nltk.corpus import brown
from nltk.corpus import gutenberg
from nltk.corpus import reuters
from nltk.tokenize import sent_tokenize
from scipy import spatial
import keras
import itertools
from keras.models import load_model

# w2v = KeyedVectors.load_word2vec_format('D:\\Downloads\\wiki.en\\wiki.en.vec')
w2v = KeyedVectors.load_word2vec_format('D:\\Downloads\\wiki.simple\\wiki.simple.vec')


def load_glove(glove_file):
    print("Loading Glove Model")
    f = open(glove_file, 'r', encoding="utf8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


# w2v = KeyedVectors.load('D:\\Downloads\\glove.6B\\glove.6B.50d.txt')
# w2v = load_glove("D:\\Downloads\\glove.6B\\glove.6B.50d.txt")
padding_len = 30
max_time_step = 100


def vectorize_sentences(sentences):
    vectorized = []
    for sentence in sentences:
        byword = sentence.split()
        concat_vector = []
        for word in byword:
            try:
                concat_vector.append(w2v[word])
            except:
                pass
        vectorized.append(concat_vector)
    return vectorized


def get_data(file_name):
    # read data from file
    # data = np.fromfile('sample_data.dat').reshape(419, 13)
    # print(np.shape(data))
    # timesteps = 3
    # dataX = []
    # for i in range(len(data) - timesteps - 1):
    #     x = data[i:(i + timesteps), :]
    #     dataX.append(x)
    # return np.array(dataX)
    with open(file_name, "r", encoding="utf8") as fp:
        sentences = fp.read().split("\n")
        vectorized = vectorize_sentences(sentences)
        return pad_sequences(vectorized, maxlen=padding_len)


# 应该是vae训练阶段输入了两个paraphrase，然后generator阶段先进行了LSTM，然后再进行decoder

if __name__ == "__main__":
    original = get_data("test_source.txt")
    paraphrase = get_data("test_target.txt")
    print(original.shape)
    input_dim = original.shape[-1]  # 13
    time_steps = original.shape[1]  # 3
    batch_size = 1

    print("time_steps:" + str(time_steps))
    print("input_dim:" + str(input_dim))
    vae, enc, gen = create_lstm_vae(input_dim,
                                    timesteps=time_steps,
                                    batch_size=batch_size,
                                    intermediate_dim=32,
                                    latent_dim=30,
                                    epsilon_std=1.,
                                    max_time_step=max_time_step)

    vae.fit([original, paraphrase], original, epochs=20)

    vae.save_weights("vae_model.h5")
    enc.save_weights("enc.h5")
    gen.save_weights("gen.h5")


    # print("start loading weights")
    # vae.load_weights("vae_model.h5")
    # enc.load_weights("enc.h5")
    # gen.load_weights("gen.h5")
    # print("loading weights finished")


    # print("predict started")
    # preds = enc.predict([original, original], batch_size=batch_size)
    # print("predict finished")
    # # pick a column to plot.
    # print("[plotting...]")
    # print("x: %s, preds: %s" % (original.shape, preds.shape))
    # plt.plot(original[:, 0, 3], label='data')
    # plt.plot(preds[:, 0, 3], label='predict')
    # plt.legend()
    # plt.show()


    # some matrix magic
    def sent_parse(sentence, mat_shape):
        data_concat = []
        word_vecs = vectorize_sentences(sentence)
        for x in word_vecs:
            data_concat.append(list(itertools.chain.from_iterable(x)))
        zero_matr = np.zeros(mat_shape)
        zero_matr[0] = np.array(data_concat)
        return zero_matr


    # input: original dimension sentence vector
    # output: text
    def print_sentence_with_w2v(sent_vect):
        word_sent = ''
        tocut = sent_vect[0]
        print(tocut.shape)
        # for i in range(int(len(sent_vect) / 300)):
        #     word_sent += w2v.most_similar(positive=[tocut[:300]], topn=1)[0][0]
        #     word_sent += ' '
        #     tocut = tocut[300:]
        for vec in tocut:
            word_sent += w2v.most_similar(positive=[vec], topn=1)[0][0]
            word_sent += " "
            # for i in sent_vect:
            #     print(i)
        print(word_sent)


    # input: encoded sentence vector
    # output: encoded sentence vector in dataset with highest cosine similarity
    def find_similar_encoding(sent_vect):
        all_cosine = []
        for sent in sent_encoded:
            result = 1 - spatial.distance.cosine(sent_vect, sent)
            all_cosine.append(result)
        data_array = np.array(all_cosine)
        maximum = data_array.argsort()[-3:][::-1][1]
        new_vec = sent_encoded[maximum]
        return new_vec


    # input: two points, integer n
    # output: n equidistant points on the line between the input points (inclusive)
    def shortest_homology(point_one, point_two, num):
        dist_vec = point_two - point_one
        sample = np.linspace(0, 1, num, endpoint=True)
        hom_sample = []
        for s in sample:
            hom_sample.append(point_one + s * dist_vec)
        return hom_sample


    # input: two written sentences, VAE batch-size, dimension of VAE input
    # output: the function embeds the sentences in latent-space, and then prints their generated text representations
    # along with the text representations of several points in between them
    def sent_2_sent(sent1, sent2, batch, dim):
        a = sent_parse([sent1], (batch, dim))
        b = sent_parse([sent2], (batch, dim))
        encode_a = enc.predict(a, batch_size=batch)
        encode_b = enc.predict(b, batch_size=batch)
        test_hom = shortest_homology(encode_a[0], encode_b[0], 5)

        for point in test_hom:
            p = gen.predict(np.array([point]))[0]
            print_sentence_with_w2v(p)


    print("encoding start")
    sent_encoded = enc.predict([original[:1], original[:1]], batch_size=1)
    print(sent_encoded.shape)
    print("encoding finished")

    print("decoding start")
    sent_decoded = gen.predict([sent_encoded, original[:1]])
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
