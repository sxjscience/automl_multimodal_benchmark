import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

seed = 123

all_train_data = pd.read_csv('machine_hack_product_sentiment/all_train.csv')


train_data, dev_data = train_test_split(all_train_data,
                                        test_size=0.2,
                                        stratify=all_train_data['Product_Type'],
                                        random_state=np.random.RandomState(seed))
train_data.to_csv(os.path.join('machine_hack_product_sentiment', 'train.csv'))
dev_data.to_csv(os.path.join('machine_hack_product_sentiment', 'dev.csv'))
print(f'#Train={len(train_data)}, #Dev={len(dev_data)}')
