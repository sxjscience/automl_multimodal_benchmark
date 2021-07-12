"""
Preprocessing Fake Job Postings data:
https://www.kaggle.com/shivamb/real-or-fake-fake-job-posting-prediction
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

directory = 'fake-job-postings/'
output_subdir = 'processed/' # where to write files
filename = 'fake_job_postings.csv'
problem_type = 'binary'

label = 'fraudulent'

seed = 123
train_name = 'train.csv'
test_name = 'test.csv'

all_train_data = pd.read_csv(directory+filename)

# check labels:
all_train_data[label].value_counts()
print("missing label values: ", np.sum(all_train_data[label].isna()))

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
