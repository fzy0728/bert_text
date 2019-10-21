import json
import numpy as np

import keras.backend as K
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from gradient_reversal import GradientReversal
from math import exp
import re, os
import codecs

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam, Adadelta

from keras.callbacks import *

from tokenizer import OurTokenizer
from data_processing import data_generator
from utils import (
    seq_pedding,
    get_category_embedding
)


os.environ['CUDA_VISIBLE_DEVICES'] = '4'

category_list = ['breast_cancer', 'hepatitis', 'diabetes', 'hypertension', 'aids']

token_dict = {}

category_ids = {i:index for index,i in enumerate(category_list)}


class bert_extend:

    def __init__(self, config_path, checkpoint_path, vocab_file):
        self.config = config_path
        self.checkpoint_path = checkpoint_path
        with codecs.open(vocab_file, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        self.tokenizer = OurTokenizer(token_dict)
        self.n_timesteps = 500

    def create_loss_weights(self):
        """Create loss weights that increase exponentially with time.

        Returns
        -------
        type : list
            A list containing a weight for each timestep.
        """
        weights = []
        for t in range(self.n_timesteps):
            weights.append(exp(-(self.n_timesteps - t)))
        return weights

    def build_bert_model(self):
        bert_model = load_trained_model_from_checkpoint(self.config, self.checkpoint_path)
        for l in bert_model.layers:
            l.trainable = True
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x3_in = Input(shape=(5,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)

        x = concatenate([x, x3_in], axis=-1)

        p = Dense(1, activation='sigmoid')(x)

        self.model = Model([x1_in, x2_in, x3_in], p)


    def sample_compile_model(self):
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(1e-5),
            metrics=['accuracy']
        )
        self.model.summary()

        return self.model

    def compile_model(self, loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=None, loss_weights=None):
        """Compile the model.

        Parameters
        ----------
        loss : str or custom loss function, optional
            Loss function to use for the training. Categorical crossentropy by default.
        optimizer : str or custom optimizer object, optional
            Optimizer to use for the training. Adam by default.
        metrics : list
            Metric to use for the training. Can be a custom metric function.
        loss_weights: dict
            Dictionary of loss weights. The items of the dictionary can be lists, with one weight per timestep.

        Returns
        -------
        type : keras.Model
            The compiled model.
        """
        if metrics is None:
            metrics = ['accuracy']
        if loss_weights is None:
            weights = self.create_loss_weights()
            loss_weights = {'domain_classifier': weights, 'aux_classifier': weights}

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics, sample_weight_mode='temporal',
                           loss_weights=loss_weights)

        return self.model

    def build_bert_domain_model(self):
        bert_model = load_trained_model_from_checkpoint(self.config, self.checkpoint_path)
        for l in bert_model.layers:
            l.trainable = True
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        # x3_in = Input(shape=(5,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)

        des = Dense(256, activation='relu')(x)
        p2 = Dense(5, activation='softmax')(des)

        flip_layer = GradientReversal(0.31)
        p_in = flip_layer(des)
        p = Dense(1, activation='sigmoid')(p_in)

        self.model = Model([x1_in, x2_in], [p, p2])
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(1e-5),
            metrics=['accuracy']
        )
        self.model.summary()
        return self.model

    def model_fit(self, train_D, valid_D):
        checkpointer = ModelCheckpoint(filepath="./checkpoint_bert.hdf5",
                                       monitor='val_acc', verbose=True, save_best_only=True, mode='auto')

        early = EarlyStopping(monitor='val_acc', patience=4, verbose=0, mode='auto')
        reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

        self.model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=10,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[reducelr, checkpointer, early],
            verbose=True
        )

    def test(self):
        pass

    def train(self, train_file):
        train = pd.read_csv(train_file)

        train_1 = train['question1'].values
        train_2 = train['question2'].values
        train_3 = train['category'].values

        labels = train['label'].astype(int).values

        data = []
        for i in zip(train_1, train_2, train_3, labels):
            data.append(i)

        # 9: 1分割数据
        random_order = list(range(len(data)))
        np.random.shuffle(random_order)
        train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
        val_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

        train_D = data_generator(train_data, self.tokenizer)
        valid_D = data_generator(val_data, self.tokenizer)

        checkpointer = ModelCheckpoint(filepath="./checkpoint_bert.hdf5",
                                       monitor='val_acc', verbose=True, save_best_only=True, mode='auto')

        early = EarlyStopping(monitor='val_acc', patience=4, verbose=0, mode='auto')
        reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
        model = self.build_bert_model()

        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=10,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[reducelr, checkpointer, early],
            verbose=True
        )

    def write_test_result(self):
        result = self.test_res()

        res = []
        for index, i in enumerate(result):
            if i > 0.5:
                res.append([index, 1])
            else:
                res.append([index, 0])
        res = pd.DataFrame(res, columns=['id', 'label'])
        res.to_csv('baseline_bert_2.csv', index=False)

    def test_res(self, test_file):
        test = pd.read_csv(test_file)
        test_1 = test['question1'].values
        test_2 = test['question2'].values
        test_3 = test['category'].values
        test_data = [i for i in zip(test_1, test_2, test_3)]
        self.model.load_weights('./checkpoint_bert1.hdf5')
        result = self.model.predict(self.t_encode(test_data), verbose=True)
        return result

    def t_encode(self, d):
        x1, x2, x3 = [], [], []
        for i in d:
            first = i[0]
            second = i[1]
            indices, segements = self.tokenizer.encode(first=first, second=second)
            cata = get_category_embedding(i[2])
            x1.append(indices)
            x2.append(segements)
            x3.append(cata)
        return [seq_pedding(x1), seq_pedding(x2), seq_pedding(x3)]



if __name__ == '__main__':
    maxlen = 100
    config_path = '/home/fuziyu/share/bert_origin/chinese/bert_config.json'
    checkpoint_path = '/home/fuziyu/share/bert_origin/chinese/bert_model.ckpt'
    dict_path = '/home/fuziyu/share/bert_origin/chinese/vocab.txt'
    s = bert_extend(config_path, checkpoint_path, dict_path)
    # s.train('./data/train.csv')
    # s.test_res('./data/dev_id.csv')
    s.build_bert_model()
    s.model.summary()
