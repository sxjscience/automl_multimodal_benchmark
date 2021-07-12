"""
Preprocessing JC Penney data (jc_penney_products):
https://www.kaggle.com/PromptCloudHQ/all-jc-penny-products

We predict sale price.
Preprocessing: remove missing sale price everywhere and other poorly curated columns.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

directory = 'jcpenney'
output_subdir = 'processed/' # where to write files
filename = 'jcpenney_com-ecommerce_sample.csv'
problem_type = 'regression'

label = 'sale_price'
eval_metric='r2'

seed = 123
train_name = 'train.csv'
test_name = 'test.csv'

all_train_data = pd.read_csv(directory+filename)

# check labels:
all_train_data[label].value_counts()
print("missing label values: ", np.sum(all_train_data[label].isna()))

# check other columns:
for col in all_train_data.columns:
    print(f"{col} # missing: {all_train_data[col].isna().sum()}")

print("out of total rows: ", len(all_train_data))

# convert numeric label column to numeric dtype:
all_train_data[label] = pd.to_numeric(all_train_data[label], errors='coerce')

# delete bad columns:
bad_columns = ['uniq_id', 'sku', 'product_image_urls','product_url', 'category', 'category_tree', 'list_price', 'Reviews']
all_train_data.drop(columns=bad_columns, inplace=True)

# delete bad rows:
keep_ind = ~all_train_data[label].isna()
all_train_data = all_train_data[keep_ind]

# extract rating from text:
bad_substr = ' out of 5'
all_train_data['average_product_rating'] = all_train_data['average_product_rating'].astype('string')
all_train_data['average_product_rating'] = all_train_data['average_product_rating'].apply(lambda x : x if pd.isna(x) else x[:-len(bad_substr)])

# convert numeric columns to numeric dtype:
numeric_features = ['total_number_reviews', 'average_product_rating']

for feat in numeric_features:
    all_train_data[feat] = pd.to_numeric(all_train_data[feat], errors='coerce')


# Split data:
train_data, test_data = train_test_split(all_train_data,
                                        test_size=0.2,
                                        random_state=np.random.RandomState(seed))

train_data.to_csv(os.path.join(directory, output_subdir, train_name), index=False)
test_data.to_csv(os.path.join(directory, output_subdir, test_name), index=False)
print(f'#Train={len(train_data)}, #Dev={len(test_data)}')


# Test run autogluon:
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon import TabularPrediction as task
from sklearn.feature_extraction.text import CountVectorizer
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

MAX_NGRAM = 300
time_limit = 300
feature_generator = AutoMLPipelineFeatureGenerator(vectorizer=CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=MAX_NGRAM, dtype=np.uint8))

predictor = TabularPredictor(label=label, path=directory+output_subdir, problem_type=problem_type, eval_metric=eval_metric)
predictor.fit(train_data, time_limit=time_limit, feature_generator=feature_generator, hyperparameters={'GBM':{}})
predictor.evaluate(test_data)


# Compute checksum:
from auto_mm_bench.utils import sha1sum
print("Train hash:\n", sha1sum(os.path.join(directory, output_subdir, train_name)))
print("Test hash:\n", sha1sum(os.path.join(directory, output_subdir, test_name)))


