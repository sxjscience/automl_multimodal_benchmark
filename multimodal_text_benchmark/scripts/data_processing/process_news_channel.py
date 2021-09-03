"""
Preprocessing Online News data for channel classification.
https://archive.ics.uci.edu/ml/datasets/online+news+popularity

Note you must first have run the processing script to produce the popularity dataset (process_news_popularity.py) before executing this script.
This script depends on the 'article_titles.p' pickle and 'OnlineNewsPopularity.csv' files produced during the popularity data preprocessing.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle

directory = 'news_channel/'
old_directory = 'news_popularity/'  # where 'article_titles.p' and 'OnlineNewsPopularity.csv' were stored by process_news_popularity.py
filename = 'OnlineNewsPopularity.csv'
label = 'channel'

problem_type = 'multiclass'
eval_metric='acc'

seed = 123
output_subdir = 'processed/' # where to write files
train_name = 'train.csv'
test_name = 'test.csv'

all_train_data = pd.read_csv(old_directory+filename)

article_titles = pickle.load(open(os.path.join(old_directory, output_subdir,"article_titles.p"), "rb" ) )

all_train_data['article_title'] = article_titles



channels = [' data_channel_is_lifestyle', ' data_channel_is_entertainment', ' data_channel_is_bus', ' data_channel_is_socmed', ' data_channel_is_tech', ' data_channel_is_world']

channel_ohe = all_train_data[channels]

def get_channel(row):
    for c in channel_ohe.columns:
        if row[c]==1:
            return c

channel_data = channel_ohe.apply(get_channel, axis=1)
all_train_data[label] = channel_data

# check labels:
all_train_data[label].value_counts()
print("missing label values: ", np.sum(all_train_data[label].isna()))



# Keep rows
keep_ind = (all_train_data['article_title'] != '') & ~all_train_data[label].isna()
all_train_data = all_train_data[keep_ind]

# Keep columns:
bad_columns = ['url', ' timedelta', ' n_tokens_title', ' title_subjectivity', ' title_sentiment_polarity',
               ' abs_title_subjectivity', ' abs_title_sentiment_polarity',
              ' avg_positive_polarity', ' min_positive_polarity', ' max_positive_polarity', ' avg_negative_polarity', ' min_negative_polarity', ' max_negative_polarity',
               ' kw_min_min', ' kw_max_min', ' kw_avg_min',
               ' kw_min_max', ' kw_max_max', ' kw_avg_max', ' kw_min_avg',
               ' kw_max_avg', ' kw_avg_avg', ' self_reference_min_shares',
               ' self_reference_max_shares', ' self_reference_avg_sharess',
               ' weekday_is_monday', ' weekday_is_tuesday', ' weekday_is_wednesday',
               ' weekday_is_thursday', ' weekday_is_friday', ' weekday_is_saturday',
               ' weekday_is_sunday', ' is_weekend', ' LDA_00', ' LDA_01', ' LDA_02',
               ' LDA_03', ' LDA_04', ' shares',
               ' data_channel_is_lifestyle', ' data_channel_is_entertainment',
               ' data_channel_is_bus', ' data_channel_is_socmed', ' data_channel_is_tech', ' data_channel_is_world',
               ]
all_train_data.drop(columns=bad_columns, inplace=True)


# Split
train_data, test_data = train_test_split(all_train_data,
                                        test_size=0.2,
                                        stratify=all_train_data[label],
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

predictor = TabularPredictor(label=label, path=directory+output_subdir, problem_type=problem_type)
predictor.fit(train_data, time_limit=time_limit, feature_generator=feature_generator, hyperparameters={'GBM':{}})
predictor.evaluate(test_data)

# Compute checksum:
from auto_mm_bench.utils import sha1sum

print("Train hash:\n", sha1sum(os.path.join(directory, output_subdir, train_name)))
print("Test hash:\n", sha1sum(os.path.join(directory, output_subdir, test_name)))