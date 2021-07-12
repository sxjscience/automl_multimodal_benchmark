"""
Preprocessing Online News Popularity data.
https://archive.ics.uci.edu/ml/datasets/online+news+popularity
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

directory = 'news_popularity/'
filename = 'OnlineNewsPopularity.csv'
problem_type = 'regression'
label = 'log_shares'

seed = 123
output_subdir = 'processed/' # where to write files
train_name = 'train.csv'
test_name = 'test.csv'

all_train_data = pd.read_csv(directory+filename)

### Need to use web scraper to get the article titles from their URLs:
import requests
from bs4 import BeautifulSoup
import time
import grequests
import pickle

num_scrape = len(all_train_data)
article_titles = [''] * num_scrape
urls = list(all_train_data['url'][:num_scrape])
outer_loop = 400
j = 0
upper_index = 0
while upper_index < num_scrape:
    lower_index = j*outer_loop
    upper_index = min(lower_index + outer_loop, num_scrape)
    urls_j = list(urls[lower_index:upper_index])
    rs = (grequests.get(u, timeout=0.5) for u in urls_j)
    for inner_i, resp in enumerate(grequests.map(rs, size=100)):
        i = lower_index + inner_i
        if resp is not None:
            soup = BeautifulSoup(resp.text, 'html.parser')
            titles = soup.find_all('title')
            if titles is not None and len(titles) > 0 and titles[0] is not None and len(titles[0]) > 0:
                try:
                    article_titles[i] = titles[0].text
                except Exception as e:
                    print(f"could not parse: {urls[i]}")
            else:
                print(f"could not parse: {urls[i]}")
        else:
            print(f"could not parse: {urls[i]}")
        if i % 500 == 0:
            print(f"processed {i} URLS of {len(all_train_data)}")
    print(f"j= {j} out of num_scrape/outer_loop")
    time.sleep(0.1)
    pickle.dump( article_titles, open( os.path.join(directory, output_subdir,"article_titles.p"), "wb" ) )
    j += 1

good_titles = [x for x in article_titles if x != '']
print(len(good_titles))
# article_titles2 = pickle.load(open(os.path.join(directory, output_subdir,"article_titles.p"), "rb" ) )
## Done Scraping ##

all_train_data['article_title'] = article_titles

# check labels:
all_train_data.rename(columns={' shares': label}, inplace=True)
all_train_data[label].value_counts()
print("missing label values: ", np.sum(all_train_data[label].isna()))

# log-transform label:
all_train_data[label] = np.log(all_train_data[label] + 1)

# check other columns:
print(all_train_data.dtypes)

for col in all_train_data.columns:
    print(f"{col} # missing: {all_train_data[col].isna().sum()}")

print("out of total rows: ", len(all_train_data))


# delete bad columns:
bad_columns = ['url', ' n_tokens_title', ' title_subjectivity', ' title_sentiment_polarity',
               ' abs_title_subjectivity', ' abs_title_sentiment_polarity']
all_train_data.drop(columns=bad_columns, inplace=True)

# delete bad rows:
keep_ind = all_train_data['article_title'] != ''
all_train_data = all_train_data[keep_ind]

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

predictor = TabularPredictor(label=label, path=directory+output_subdir, problem_type=problem_type)
predictor.fit(train_data, time_limit=time_limit, feature_generator=feature_generator, hyperparameters={'GBM':{}})
predictor.evaluate(test_data)


# Compute checksum:
from auto_mm_bench.utils import sha1sum
print("Train hash:\n", sha1sum(os.path.join(directory, output_subdir, train_name)))
print("Test hash:\n", sha1sum(os.path.join(directory, output_subdir, test_name)))
