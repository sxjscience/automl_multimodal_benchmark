import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.core.utils.files import download
import numpy as np
import gzip
import random
import ast
import os

DOWNLOAD_DIR = 'airbnb_data'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_raw_airbnb(region='united-states/wa/seattle', date='2020-10-25'):

    url_prefix = f'http://data.insideairbnb.com/{region}/{date}/data/'
    listing_csv_url = url_prefix + 'listings.csv.gz'
    out_path = download(listing_csv_url, path=os.path.join(DOWNLOAD_DIR,
                                                           region.replace('/', '_') + date + '.csv'))
    return out_path

rng = np.random.RandomState(123)
all_data = pd.read_csv('cleansed_listings_dec18.csv', low_memory=False)
all_verifications = []
for verification in all_data['host_verifications']:
    dat = ast.literal_eval(verification)
    if dat is not None:
        all_verifications.extend(dat)
all_verifications_set = set(all_verifications)
additional_cols = {f'host_verifications_{method}':
                       np.zeros((len(all_data), ), dtype=np.bool)
                   for method in all_verifications_set}

sorted_price = sorted(all_data['price'])
boundaries = []

for i in range(1, 10):
    boundaries.append(sorted_price[int(i / 11 * len(sorted_price))])


all_data['price_label'] = np.searchsorted(boundaries,
                                          all_data['price'], 'right')

for i, verification in enumerate(all_data['host_verifications']):
    dat = ast.literal_eval(verification)
    if dat is not None:
        for method in dat:
            additional_cols[f'host_verifications_{method}'][i] = True
all_data.drop('host_verifications', axis=1)
for name, value in additional_cols.items():
    all_data[name] = value


train_data, test_data = train_test_split(all_data,
                                         test_size=0.2,
                                         shuffle=True,
                                         stratify=all_data['price_label'],
                                         random_state=rng)

train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)


base_dir = 'airbnb_melbourne'
os.makedirs(base_dir, exist_ok=True)
train_data.to_parquet(os.path.join(base_dir, 'train.pq'))
test_data.to_parquet(os.path.join(base_dir, 'test.pq'))
print('#Train', len(train_data), ', #Test', len(test_data))
