import os
import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
        description='Process the Data Scientist Salary Prediction dataset from MachineHack.')
parser.add_argument('--dir_path', type=str)
args = parser.parse_args()

seed = 123
train_path = os.path.join(args.dir_path, 'Data', 'Final_Train_Dataset.csv')
test_path = os.path.join(args.dir_path, 'Data', 'Final_Test_Dataset.csv')

all_train_df = pd.read_csv(train_path, index_col=0)
competition_df = pd.read_csv(test_path, index_col=None)

all_train_df.drop('company_name_encoded', axis=1, inplace=True)
competition_df.drop('company_name_encoded', axis=1, inplace=True)

train_df, test_df = train_test_split(all_train_df, test_size=0.2, random_state=np.random.RandomState(seed))
train_df.to_csv(os.path.join(args.dir_path, 'train.csv'), index=False)
test_df.to_csv(os.path.join(args.dir_path, 'test.csv'), index=False)
competition_df.to_csv(os.path.join(args.dir_path, 'competition.csv'), index=False)
print(f'#Train={len(train_df)}, #Dev={len(test_df)}, #Competition={len(competition_df)}')
