import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

all_data = pd.read_csv('jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test_data = pd.read_csv('jigsaw-unintended-bias-in-toxicity-classification/test_private_expanded.csv')

all_data['target'] = all_data['target'] > 0.5

rng = np.random.RandomState(123)
train_data, valid_data = train_test_split(all_data, test_size=0.2, shuffle=True,
                                          random_state=rng, stratify=all_data['target'])
train_data.reset_index(drop=True, inplace=True)
valid_data.reset_index(drop=True, inplace=True)


base_dir = 'jigsaw_unintended_bias'
os.makedirs(base_dir, exist_ok=True)
train_data.to_parquet(os.path.join(base_dir, 'train.pq'))
valid_data.to_parquet(os.path.join(base_dir, 'dev.pq'))
test_data.to_parquet(os.path.join(base_dir, 'test.pq'))
print('#Train', len(train_data), ', #Dev', len(valid_data), ', #Test', len(test_data))
