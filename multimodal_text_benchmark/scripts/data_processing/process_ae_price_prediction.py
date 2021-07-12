import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

all_data = pd.read_csv('ae_com.csv')

price_l = []
for price in all_data['price']:
    post_fix = ' USD'
    price_l.append(float(price[:-len(post_fix)]))

all_data['price'] = price_l
train_df, test_df = train_test_split(all_data, test_size=0.2)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


base_dir = 'ae_price_prediction'
os.makedirs(base_dir, exist_ok=True)
train_df.to_parquet(os.path.join(base_dir, 'train.pq'))
test_df.to_parquet(os.path.join(base_dir, 'test.pq'))
print('#Train', len(train_df), ', #Test', len(test_df))
