import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('mercari-price-suggestion-challenge/train.tsv', sep='\t')
test_data = pd.read_csv('mercari-price-suggestion-challenge/test.tsv', sep='\t')

train_cat1 = []
train_cat2 = []
train_cat3 = []

test_cat1 = []
test_cat2 = []
test_cat3 = []
train_category_name = train_data['category_name']
test_category_name = test_data['category_name']


for ele in train_data['category_name']:
    if isinstance(ele, str):
        categories = ele.split('/', 2)
        train_cat1.append(categories[0])
        train_cat2.append(categories[1])
        train_cat3.append(categories[2])
    else:
        train_cat1.append(None)
        train_cat2.append(None)
        train_cat3.append(None)


for ele in test_data['category_name']:
    if isinstance(ele, str):
        categories = ele.split('/', 2)
        test_cat1.append(categories[0])
        test_cat2.append(categories[1])
        test_cat3.append(categories[2])
    else:
        test_cat1.append(None)
        test_cat2.append(None)
        test_cat3.append(None)

train_data['log_price'] = np.log(train_data['price'] + 1)
train_data.drop('category_name', axis=1)
train_data['cat1'] = train_cat1
train_data['cat2'] = train_cat2
train_data['cat3'] = train_cat3

test_data.drop('category_name', axis=1)
test_data['cat1'] = test_cat1
test_data['cat2'] = test_cat2
test_data['cat3'] = test_cat3

train_df, dev_df = train_test_split(train_data, test_size=0.2)
train_df.reset_index(drop=True, inplace=True)
dev_df.reset_index(drop=True, inplace=True)


test_df = test_data

base_dir = 'mercari_price_suggestion'
os.makedirs(base_dir, exist_ok=True)
train_df.to_parquet(os.path.join(base_dir, 'train.pq'))
dev_df.to_parquet(os.path.join(base_dir, 'dev.pq'))
test_df.to_parquet(os.path.join(base_dir, 'test.pq'))
print(f'#Train={len(train_df)}, #Dev={len(dev_df)}, #Test={len(test_df)}')
