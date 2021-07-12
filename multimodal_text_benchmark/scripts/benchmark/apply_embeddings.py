import argparse
import pandas as pd
import os
import json
import numpy as np
from auto_mm_bench.datasets import dataset_registry


dataset_l = [
    "product_sentiment_machine_hack",
    "google_qa_question_type_reason_explanation",
    "google_qa_answer_type_reason_explanation",
    "women_clothing_review",
    "melbourne_airbnb",
    "ae_price_prediction",
    "mercari_price_suggestion100K",
    "jigsaw_unintended_bias100K",
    "imdb_genre_prediction",
    "fake_job_postings2",
    "kick_starter_funding",
    "jc_penney_products",
    "wine_reviews",
    "news_popularity2",
    "news_channel"
]

embedding_type = 'pretrain_text_embedding'
base_dir = 'embeddings'
out_dir = 'pre_embedding_data'

for dataset in dataset_l:
    print('Processing', dataset)
    train_dataset = dataset_registry.create(dataset, 'train')
    test_dataset = dataset_registry.create(dataset, 'test')
    train_features = np.load(os.path.join(base_dir, dataset, embedding_type,
                                          'train.npy'))
    test_features = np.load(os.path.join(base_dir, dataset, embedding_type,
                                         'test.npy'))
    with open(os.path.join(base_dir, dataset, embedding_type,
                           'text_columns.json'), 'r') as in_f:
        text_columns = json.load(in_f)
    other_columns = [col for col in train_dataset.feature_columns if col not in text_columns]
    train_data = train_dataset.data[other_columns + train_dataset.label_columns]
    test_data = test_dataset.data[other_columns + train_dataset.label_columns]
    merged_train_data = train_data.join(pd.DataFrame(
        train_features,
        columns=[f'pretrain_feature{i}' for i in range(train_features.shape[1])]))
    merged_train_data.reset_index(drop=True, inplace=True)
    merged_test_data = test_data.join(pd.DataFrame(
        test_features,
        columns=[f'pretrain_feature{i}' for i in range(test_features.shape[1])]))
    merged_test_data.reset_index(drop=True, inplace=True)
    os.makedirs(os.path.join(out_dir, dataset), exist_ok=True)
    merged_train_data.to_parquet(os.path.join(out_dir, dataset, 'train.pq'))
    merged_test_data.to_parquet(os.path.join(out_dir, dataset, 'test.pq'))
