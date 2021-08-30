import os
import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
        description='Process the Data Scientist Salary Prediction dataset from MachineHack.')
parser.add_argument('--dir_path', type=str, default='california-house-prices')
args = parser.parse_args()

seed = 123
train_path = os.path.join(args.dir_path, 'train.csv')
test_path = os.path.join(args.dir_path, 'test.csv')

all_train_df = pd.read_csv(train_path, index_col=0)
competition_df = pd.read_csv(test_path, index_col=0)

def preprocess(df, with_tax_values=True, log_scale_lot=True,
               log_scale_listed_price=True, has_label=True):
    new_df = df.copy()
    new_df['Elementary School'] = new_df['Elementary School'].apply(lambda ele: str(ele)[:-len(' Elementary School')] if str(ele).endswith('Elementary School') else ele)
    if log_scale_lot:
        new_df['Lot'] = np.log(new_df['Lot'] + 1)
    if log_scale_listed_price:
        log_listed_price = np.log(new_df['Listed Price']).clip(0, None)
        new_df['Listed Price'] = log_listed_price
    if with_tax_values:
        new_df['Tax assessed value'] = np.log(new_df['Tax assessed value'] + 1)
        new_df['Annual tax amount'] = np.log(new_df['Annual tax amount'] + 1)
    else:
        new_df.drop('Tax assessed value', axis=1, inplace=True)
        new_df.drop('Annual tax amount', axis=1, inplace=True)
    if has_label:
        new_df['Sold Price'] = np.log(new_df['Sold Price'])
    return new_df

all_train_df = preprocess(all_train_df, with_tax_values=True, has_label=True)
competition_df = preprocess(competition_df, with_tax_values=True, has_label=False)

all_train_df = preprocess(all_train_df, )

os.makedirs(os.path.join(args.dir_path, 'processed'), exist_ok=True)
train_df, test_df = train_test_split(all_train_df, test_size=0.2, random_state=np.random.RandomState(seed))
train_df.to_csv(os.path.join(args.dir_path, 'processed', 'train.csv'), index=False)
test_df.to_csv(os.path.join(args.dir_path, 'processed','test.csv'), index=False)
competition_df.to_csv(os.path.join(args.dir_path, 'processed','competition.csv'), index=False)
print(f'#Train={len(train_df)}, #Dev={len(test_df)}, #Competition={len(competition_df)}')
