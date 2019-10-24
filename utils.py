import numpy as np
import json

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

category_list = ['breast_cancer', 'hepatitis', 'diabetes', 'hypertension', 'aids']

category_ids = {i: index for index, i in enumerate(category_list)}


def seq_pedding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))])if len(x) < ML else x for x in X
    ])


def get_category_embedding(data):
    s = np.zeros(5)
    s[category_ids[data]] = 1
    return s


def word2vec_train(word2vec_file, word2vec_id):
    mention_id = json.load(open(word2vec_id))
    n_symbols = len(mention_id) + 1

    word2model = KeyedVectors.load(word2vec_file)

    mention_id = {key: value for key, value in mention_id.items() if str(value) in word2model}
    # mention_vec = {key: word2model[value] for key, value in mention_id.items()}

    embedding_weights = np.zeros((n_symbols, 200))
    for word, index in mention_id.items():
        embedding_weights[index, :] = word2model[str(index)]

    return mention_id, embedding_weights, n_symbols

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
#
if __name__ == '__main__':
    s1, s2, len_s = word2vec_train('./model/line_word2vec.pkl', './model/forum_mention_id.json')
    print(s2[1], len_s)
