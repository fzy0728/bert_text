import pandas as pd
import numpy as np
import re, os
import codecs

from tokenizer import (
    OurTokenizer,
    JiebaEmbedding
)
from data_processing import data_generator
from utils import (
    seq_pedding,
    get_category_embedding,
    word2vec_train
)

from keras.utils.np_utils import to_categorical
from build_model import BuildModel


os.environ['CUDA_VISIBLE_DEVICES'] = '4'

token_dict = {}


class bert_extend:

    def __init__(self, config_path, checkpoint_path, vocab_file, word2vec, word2id, user_dict=None):
        print('init...')
        with codecs.open(vocab_file, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        self.tokenizer = OurTokenizer(token_dict)
        # self.n_timesteps = 500
        self.model = BuildModel(config_path, checkpoint_path, 500, self.tokenizer)
        self.mention_id, self.mention_vec = word2vec_train(word2vec, word2id)
        self.jieba_embed = JiebaEmbedding(user_dict, self.mention_id)
        print('init...Done!')

    def test(self):
        pass

    def train(self, train_file):
        train = pd.read_csv(train_file)

        train_1 = train['question1'].values
        train_2 = train['question2'].values
        train_3 = train['category'].values

        labels = train['label'].astype(int).values
        labels = to_categorical(labels,2)
        labels = labels.astype(np.int32)
#         print(labels[0])

        data = []
        for i in zip(train_1, train_2, train_3, labels):
            data.append(i)

        # 9: 1分割数据
        random_order = list(range(len(data)))
        np.random.shuffle(random_order)
        train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
        val_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

        # print(val_data[0])
        train_D = data_generator(train_data, self.tokenizer)
        valid_D = data_generator(val_data, self.tokenizer)
        self.model.build_text_cnn_model()
        self.model.model_fit_2(train_D, valid_D)


    def write_test_result(self, test_file):
        result = self.test_res(test_file)

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
        # test_3 = test['category'].values
        test_data = [i for i in zip(test_1, test_2)]
        self.model.build_text_cnn_model()
        self.model.model.load_weights('./checkpoint_bert.hdf5')
        result = self.model.model.predict(self.t_encode(test_data), verbose=True)
#         print(result[0], result[1])
        res = np.argmax(result[0], axis=1)
#         print(res)
        return res

    def t_encode(self, d):
        x1, x2, x3 = [], [], []
        for i in d:
            first = i[0]
            second = i[1]
            indices, segements = self.tokenizer.encode(first=first, second=second)
            x1.append(indices)
            x2.append(segements)
        return [seq_pedding(x1), seq_pedding(x2)]


if __name__ == '__main__':
    maxlen = 100
    config_path = '/home/fuziyu/share/bert_origin/chinese/bert_config.json'
    checkpoint_path = '/home/fuziyu/share/bert_origin/chinese/bert_model.ckpt'
    dict_path = '/home/fuziyu/share/bert_origin/chinese/vocab.txt'

    word2vec_file = './model/line_word2vec.pkl'
    word2vec_id = './model/forum_mention_id.json'

    s = bert_extend(config_path, checkpoint_path, dict_path, word2vec_file, word2vec_id)
    s.train('./data/train.csv')
    s.write_test_result('./data/dev_id.csv')
#     s.build_bert_domain_model()
#     s.model.summary()
