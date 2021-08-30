import os
import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
        description='Process the Book Price Prediction dataset from MachineHack.')
parser.add_argument('--dir_path', type=str)
args = parser.parse_args()

seed = 123
train_path = os.path.join(args.dir_path, 'Participants_Data', 'Data_Train.xlsx')
test_path = os.path.join(args.dir_path, 'Participants_Data', 'Data_Test.xlsx')

all_train_df = pd.read_excel(train_path, engine='openpyxl')
competition_df = pd.read_excel(test_path, engine='openpyxl')

# Convert Reviews
all_train_df.loc[:, 'Reviews'] = pd.to_numeric(all_train_df['Reviews'].apply(
    lambda ele: ele[:-len(' out of 5 stars')]))
competition_df.loc[:, 'Reviews'] = pd.to_numeric(
    competition_df['Reviews'].apply(lambda ele: ele[:-len(' out of 5 stars')]))
# Convert Ratings
all_train_df.loc[:, 'Ratings'] = pd.to_numeric(all_train_df['Ratings'].apply(
    lambda ele: ele.replace(',', '')[:-len(' customer reviews')]))
competition_df.loc[:, 'Ratings'] = pd.to_numeric(
    competition_df['Ratings'].apply(lambda ele: ele.replace(',', '')[:-len(' customer reviews')]))
# Convert Price to log scale
all_train_df.loc[:, 'Price'] = np.log10(all_train_df['Price'] + 1)

train_df, test_df = train_test_split(all_train_df, test_size=0.2, random_state=np.random.RandomState(seed))
train_df.to_csv(os.path.join(args.dir_path, 'train.csv'), index=False)
test_df.to_csv(os.path.join(args.dir_path, 'test.csv'), index=False)
competition_df.to_csv(os.path.join(args.dir_path, 'competition.csv'), index=False)
print(f'#Train={len(train_df)}, #Dev={len(test_df)}, #Competition={len(competition_df)}')
