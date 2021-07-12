import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

all_data = pd.read_csv('google-quest-challenge/train.csv')
test_data = pd.read_csv('google-quest-challenge/test.csv')

rng = np.random.RandomState(123)
train_data, valid_data = train_test_split(all_data, test_size=0.2, shuffle=True, random_state=rng)
train_data.reset_index(drop=True, inplace=True)
valid_data.reset_index(drop=True, inplace=True)


base_dir = 'google_quest_qa'
os.makedirs(base_dir, exist_ok=True)
train_data.to_parquet(os.path.join(base_dir, 'train.pq'))
valid_data.to_parquet(os.path.join(base_dir, 'dev.pq'))
test_data.to_parquet(os.path.join(base_dir, 'test.pq'))
print('#Train', len(train_data), ', #Dev', len(valid_data), ', #Test', len(test_data))

