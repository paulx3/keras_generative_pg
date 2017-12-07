'''
@author: wanzeyu

@contact: wan.zeyu@outlook.com

@file: helper.py

@time: 2017/12/4 2:19
'''
import itertools

import numpy as np
from gensim.models import KeyedVectors

np.set_printoptions(threshold=np.nan)
END_TOKEN = np.array(300 * [1])
UNK_TOKEN = np.array(300 * [2])
ZERO_TOKEN = np.array(300 * [0])
padding_len = 20

w2v = KeyedVectors.load_word2vec_format('./wiki.simple.vec')


# some matrix magic
def sent_parse(sentence, mat_shape):
    data_concat = []
    word_vecs = vectorize_sentences(sentence)
    for x in word_vecs:
        data_concat.append(list(itertools.chain.from_iterable(x)))
    zero_matr = np.zeros(mat_shape)
    zero_matr[0] = np.array(data_concat)
    return zero_matr


# input: two points, integer n
# output: n equidistant points on the line between the input points (inclusive)
def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint=True)
    hom_sample = []
    for s in sample:
        hom_sample.append(point_one + s * dist_vec)
    return hom_sample


def load_glove(glove_file):
    print("Loading Glove Model")
    f = open(glove_file, 'r', encoding="utf8")
    model = {}
    for line in f:
        slit_line = line.split()
        word = slit_line[0]
        embedding = np.array([float(val) for val in slit_line[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def vectorize_sentences(sentences):
    vectorized = []
    for sentence in sentences:
        byword = sentence.split()
        concat_vector = []
        for word in byword:
            try:
                concat_vector.append(w2v[word])
            except:
                concat_vector.append(UNK_TOKEN)
        concat_vector.append(END_TOKEN)
        if padding_len - len(concat_vector) > 0:
            for _ in range(padding_len - len(concat_vector)):
                concat_vector.append(ZERO_TOKEN)
        else:
            concat_vector = concat_vector[:padding_len]
        concat_vector = np.array(concat_vector)
        vectorized.append(concat_vector)
    return np.array(vectorized)


def get_data(file_name):
    with open(file_name, "r", encoding="utf8") as fp:
        sentences = fp.read().split("\n")
        return vectorize_sentences(sentences)


# input: original dimension sentence vector
# output: text
def print_sentence_with_w2v(sent_vect):
    for tocut in sent_vect[:10]:
        word_sent = ''
        for vec in tocut:
            word_sent += w2v.most_similar(positive=[vec], topn=1)[0][0]
            word_sent += " "
        try:
            print(word_sent)
            print("=====================sent===============")
        except Exception as e:
            print("print exception")
