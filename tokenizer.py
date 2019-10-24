
import jieba
import numpy as np

from keras_bert import load_trained_model_from_checkpoint, Tokenizer

from keras.preprocessing import sequence

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

class JiebaEmbedding():

    def __init__(self, word2id, user_dict=None, max_len=100):

        if user_dict:
            jieba.load_userdict(user_dict)
        self.word2id = word2id
        self.max_len = max_len

    def encode(self, data):
        data_id = []
        for sentence in data:
            sentence_id = []
            for word in jieba.cut(sentence):
                try:
                    sentence_id.append(self.word2id[word])
                except:
                    sentence_id.append(0)
            data_id.append(sentence_id)
        data_id = sequence.pad_sequences(data_id, self.max_len)
        return data_id

    def cut(self, sentence, cut_all=False):
        return jieba.cut(sentence, cut_all)
