import codecs
from keras_bert import Tokenizer
dict_path = '/home/fuziyu/share/bert_origin/chinese/vocab.txt'
token_dict = {}

with codecs.open(dict_path, 'r') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


tokenizer = Tokenizer(token_dict)
print(tokenizer.tokenize('unaffable'))

indices, segments = tokenizer.encode('unaffable')
print(indices, segments)
print(tokenizer.tokenize(first='unaffable UNK', second='钢'))

indices, segments = tokenizer.encode(first='unaffable UNK', second='钢', max_len=10)
print(indices)
print(segments)


