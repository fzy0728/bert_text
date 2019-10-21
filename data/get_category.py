import pandas as pd

res = pd.read_csv('./train.csv')
category = res['category'].values
s = set(category)
print(s)

