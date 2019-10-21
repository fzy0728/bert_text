import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam, Adadelta

from keras.callbacks import *

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
maxlen = 100
config_path = '/home/fuziyu/share/bert_origin/chinese/bert_config.json'
checkpoint_path = '/home/fuziyu/share/bert_origin/chinese/bert_model.ckpt'
dict_path = '/home/fuziyu/share/bert_origin/chinese/vocab.txt'

category_list = ['breast_cancer', 'hepatitis', 'diabetes', 'hypertension', 'aids']

token_dict = {}

category_ids = {i:index for index,i in enumerate(category_list)}

def get_category_embedding(data):
    s = np.zeros(5)
    s[category_ids[data]] = 1
    return s

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

class OurTokenizer(Tokenizer):
    def _tokenize(self, text, unk_dict=None):
        R = []
        for c in text:
            if unk_dict and c in unk_dict:
                R.append('UNK')
            elif c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('UNK')
        return R

    def encode(self, first, second=None, max_len=None):
#         unk_dict = set(first) & set(second) if second is not None else None
        unk_dict = None
        first_tokens = self._tokenize(first, unk_dict)
        second_tokens = self._tokenize(second, unk_dict) if second is not None else None
        self._truncate(first_tokens, second_tokens, max_len)
        tokens, first_len, second_len = self._pack(first_tokens, second_tokens)

        token_ids = self._convert_tokens_to_ids(tokens)
        segment_ids = [0] * first_len + [1] * second_len
        if max_len is not None:
            pad_len = max_len - first_len - second_len
            token_ids += [self._pad_index] * pad_len
            segment_ids += [0] * pad_len
        return token_ids, segment_ids


tokenizer = OurTokenizer(token_dict)

def seq_pedding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))])if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            x1, x2, x3, Y = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text1 = d[0][:maxlen]
                text2 = d[1][:maxlen]
                indices, segments = tokenizer.encode(first=text1, second=text2)
                text3 = get_category_embedding(d[2])
                y = d[3]
                x1.append(indices)
                x2.append(segments)
                x3.append(text3)
                Y.append([y])
                if len(x1) == self.batch_size or i == idxs[-1]:
                    x1 = seq_pedding(x1)
                    x2 = seq_pedding(x2)
                    x3 = seq_pedding(x3)
                    Y = seq_pedding(Y)
                    yield [x1, x2, x3], Y
                    x1, x2, x3, Y = [], [], [] ,[]

def build_model():

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True
    x1_in = Input(shape=(None, ))
    x2_in = Input(shape=(None, ))
    x3_in = Input(shape=(5,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)

    x = concatenate([x, x3_in], axis=-1)

    p = Dense(1, activation='sigmoid')(x)

    model = Model([x1_in, x2_in, x3_in], p)
    model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(1e-5),
            metrics=['accuracy']
            )
    model.summary()
    return model
def train():
    train = pd.read_csv('./data/train.csv')

    train_1 = train['question1'].values
    train_2 = train['question2'].values
    train_3 = train['category'].values

# category_train = get_category_embedding(train_3)

    labels = train['label'].astype(int).values

# test_1 = test['question1'].values
# test_2 = test['question2'].values
# test_3 = test['category'].values
# category_test = get_category_embedding(test_3)

    data = []
    for i in zip(train_1, train_2, train_3, labels):
        data.append(i)

# test_data = [i for i in zip(test_1, test_2)]


# 9: 1分割数据
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
    val_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

    train_D = data_generator(train_data)
    valid_D = data_generator(val_data)
# test_D = data_generator(test_data)


    checkpointer = ModelCheckpoint(filepath="./checkpoint_bert.hdf5",
        monitor='val_acc', verbose=True, save_best_only=True, mode='auto')

    early = EarlyStopping(monitor='val_acc', patience=4, verbose=0, mode='auto')
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    model = build_model()

    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=10,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        callbacks=[reducelr, checkpointer, early],
        verbose=True
        )

def t_encode(d):
    x1, x2, x3 = [], [], []
    for i in d:
        first = i[0]
        second = i[1]
        indices, segements = tokenizer.encode(first=first, second=second)
        cata = get_category_embedding(i[2])
        x1.append(indices)
        x2.append(segements)
        x3.append(cata)
    return [seq_pedding(x1), seq_pedding(x2), seq_pedding(x3)]

def test_res():
    test = pd.read_csv('./data/dev_id.csv')
    test_1 = test['question1'].values
    test_2 = test['question2'].values
    test_3 = test['category'].values
    test_data = [i for i in zip(test_1, test_2, test_3)]
    model = build_model()
    model.load_weights('./checkpoint_bert1.hdf5')
    result = model.predict(t_encode(test_data), verbose=True)
    return result

# tarin_D = data_generator(train_data)
def test():
    result = test_res()

    res = []
    for index, i in enumerate(result):
        if i > 0.5:
            res.append([index, 1])
        else:
            res.append([index, 0])
    res = pd.DataFrame(res, columns=['id', 'label'])
    res.to_csv('baseline_bert_2.csv', index=False)

# tarin_D = data_generator(train_data)
if __name__ == '__main__':
    train()
    test()
