import numpy as np


from utils import (
    seq_pedding,
    get_category_embedding
)


class data_generator:
    def __init__(self, data, tokenizer, batch_size=32, maxlen=100):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        self.maxlen = maxlen
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.tokenizer = tokenizer

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            x1, x2, Y, Y1 = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text1 = d[0][:self.maxlen]
                text2 = d[1][:self.maxlen]
                indices, segments = self.tokenizer.encode(first=text1, second=text2)
                text3 = get_category_embedding(d[2])
#                 print(text3)
                y = d[3]
                x1.append(indices)
                x2.append(segments)
                Y1.append(text3)
                Y.append(y)
                if len(x1) == self.batch_size or i == idxs[-1]:
                    x1 = seq_pedding(x1)
                    x2 = seq_pedding(x2)
                    Y1 = seq_pedding(Y1)
                    Y = seq_pedding(Y)
                    yield [x1, x2], [Y, Y1]
                    x1, x2, Y, Y1 = [], [], [] ,[]
