import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from auto_mm_bench.utils import download

os.makedirs('women_clothing_review', exist_ok=True)
download('s3://automl-mm-bench/women_clothing_review/women_clothing_review_all.pq',
         'women_clothing_review/women_clothing_review_all.pq')
all_data = pd.read_parquet('women_clothing_review/women_clothing_review_all.pq')

rng = np.random.RandomState(123)
# all_data['Division Name'].fillna('None', inplace=True)
# all_data['Department Name'].fillna('None', inplace=True)
# all_data['Class Name'].fillna('None', inplace=True)

train_data, test_data = train_test_split(
    all_data,
    test_size=0.2, shuffle=True,
    random_state=rng,
    stratify=all_data['Division Name'].fillna('None').astype('category').cat.codes)
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

train_data.to_parquet(os.path.join('women_clothing_review', 'train.pq'))
test_data.to_parquet(os.path.join('women_clothing_review', 'test.pq'))
print(len(train_data))
print(len(test_data))
