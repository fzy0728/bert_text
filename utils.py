import numpy as np

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