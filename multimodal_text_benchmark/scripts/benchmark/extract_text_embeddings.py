from autogluon.text import TextPredictor
from autogluon.text.text_prediction.mx.models import MultiModalTextModel
import boto3
import os
import argparse
import sklearn
import json
import pandas as pd
import numpy as np
from autogluon.text import TextPredictor, ag_text_presets
from autogluon.text.text_prediction.infer_types import infer_column_problem_types

from auto_mm_bench.datasets import dataset_registry


parser = argparse.ArgumentParser('Extract embeddings from the input dataframe.')
parser.add_argument('--dataset', default=None)
parser.add_argument('--stat_df', help='Store the boto3 path statistics.')
parser.add_argument('--model_name', help='Name of the model',
                    default='ag_text_multimodal-electra_base_late_fusion_concat_e10_avg3-no')
parser.add_argument('--extract_pretrained', action='store_true')
parser.add_argument('--verify_performance', action='store_true')
parser.add_argument('--use_file_name', action='store_true')
parser.add_argument('--model_download_path',
                    default='model_download',
                    help='Path to download the model.')

MULTIMODAL_TEXT_MODEL_NAME = 'ag_text_multimodal-electra_base_late_fusion_concat_e10_avg3-no'
TEXT_MODEL_NAME = 'ag_text_only-electra_base_all_text_e10_avg3-no'

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


def extract_pretrained_embedding(dataset):
    hyperparameters = ag_text_presets.create('default')
    hyperparameters['models']['MultimodalTextModel']['search_space'][
        'model.num_trainable_layers'] = 0
    hyperparameters['models']['MultimodalTextModel']['search_space'][
        'model._disable_update'] = True
    hyperparameters['models']['MultimodalTextModel']['search_space'][
        'optimization.num_train_epochs'] = 1
    hyperparameters['models']['MultimodalTextModel']['search_space'][
        'preprocessing.categorical.convert_to_text'] = True
    hyperparameters['models']['MultimodalTextModel']['search_space']['optimization.lr'] = 0.
    seed = 123
    train_dataset = dataset_registry.create(dataset, 'train')
    test_dataset = dataset_registry.create(dataset, 'test')
    train_data1, tuning_data1 = sklearn.model_selection.train_test_split(
        train_dataset.data,
        test_size=0.05,
        random_state=np.random.RandomState(seed))
    column_types, inferred_problem_type = infer_column_problem_types(train_data1,
                                                                     tuning_data1,
                                                                     label_columns=train_dataset.label_columns,
                                                                     problem_type=train_dataset.problem_type)
    text_feature_columns = [col_name for col_name in train_dataset.feature_columns if
                            column_types[col_name] == 'text']
    train_text_only_data = train_dataset.data[text_feature_columns + train_dataset.label_columns]
    test_text_only_data = test_dataset.data[text_feature_columns + test_dataset.label_columns]
    sampled_train_data = train_text_only_data.sample(10)
    predictor = TextPredictor(label=train_dataset.label_columns)
    predictor.fit(train_data=sampled_train_data,
                  column_types=column_types,
                  hyperparameters=hyperparameters)
    train_features = predictor.extract_embedding(train_text_only_data)
    test_features = predictor.extract_embedding(test_text_only_data)
    save_base_dir = f'embeddings/{dataset}/pretrain_text_embedding'
    os.makedirs(save_base_dir, exist_ok=True)
    np.save(os.path.join(save_base_dir, 'train.npy'), train_features)
    np.save(os.path.join(save_base_dir, 'test.npy'), test_features)
    with open(os.path.join(save_base_dir, 'text_columns.json'), 'w') as in_f:
        json.dump(text_feature_columns, in_f)


def extract_finetuned_embedding(dataset, stat_df, verify_performance):
    model_l = stat_df['Unnamed: 0']
    multimodal_idx = None
    text_only_idx = None
    for i, model in enumerate(model_l):
        if model == MULTIMODAL_TEXT_MODEL_NAME:
            multimodal_idx = i
        elif model == TEXT_MODEL_NAME:
            text_only_idx = i
    if multimodal_idx is None or text_only_idx is None:
        raise NotImplementedError("Model not found!")

    multimodal_model_remote_path = stat_df[dataset].iloc[multimodal_idx]
    text_model_remote_path = stat_df[dataset].iloc[text_only_idx]
    postfix = '/test_score.json'

    multimodal_remote_dir_name = multimodal_model_remote_path[:-len(postfix)]
    text_remote_dir_name = text_model_remote_path[:-len(postfix)]
    print(multimodal_remote_dir_name)


    def downloadDirectoryFroms3(bucketName, remoteDirectoryName, local_dir_path):
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(bucketName)
        for obj in bucket.objects.filter(Prefix=remoteDirectoryName):
            print(obj.key)
            download_path = os.path.join(local_dir_path, obj.key)
            if not os.path.exists(os.path.dirname(download_path)):
                os.makedirs(os.path.dirname(download_path), exist_ok=True)
            bucket.download_file(obj.key, download_path)


    downloadDirectoryFroms3('automl-mm-bench', multimodal_remote_dir_name, args.model_download_path)
    downloadDirectoryFroms3('automl-mm-bench', text_remote_dir_name, args.model_download_path)

    # Multimodal Embedding
    multimodal_text_nn = MultiModalTextModel.load(os.path.join(args.model_download_path,
                                                               multimodal_remote_dir_name,
                                                               'saved_model'))
    print(multimodal_text_nn)
    with open(os.path.join(args.model_download_path, multimodal_remote_dir_name,
                           'test_score.json'), 'r') as in_f:
        model_test_score = json.load(in_f)
        multimodal_loaded_score_val = list(model_test_score.values())[0]
    train_dataset = dataset_registry.create(dataset, 'train')
    test_dataset = dataset_registry.create(dataset, 'test')
    train_features = multimodal_text_nn.extract_embedding(train_dataset.data)
    test_features = multimodal_text_nn.extract_embedding(test_dataset.data)
    if verify_performance:
        multimodal_pred_score = multimodal_text_nn.evaluate(test_dataset.data)
        assert multimodal_pred_score == multimodal_loaded_score_val,\
            f"MultiModalText NN: Predicted score={multimodal_pred_score}, " \
            f"Loaded score={multimodal_loaded_score_val}, " \
            f"Dataset={dataset}"
    os.makedirs(f'embeddings/{dataset}/multimodal_embedding', exist_ok=True)
    np.save(os.path.join(f'embeddings/{dataset}/multimodal_embedding', 'train.npy'), train_features)
    np.save(os.path.join(f'embeddings/{dataset}/multimodal_embedding', 'test.npy'), test_features)

    # Text Embedding
    text_nn = MultiModalTextModel.load(os.path.join(args.model_download_path,
                                                    text_remote_dir_name, 'saved_model'))
    with open(os.path.join(args.model_download_path, text_remote_dir_name,
                           'test_score.json'), 'r') as in_f:
        model_test_score = json.load(in_f)
        text_loaded_score_val = list(model_test_score.values())[0]
    train_text_embeddings = text_nn.extract_embedding(train_dataset.data)
    test_text_embeddings = text_nn.extract_embedding(test_dataset.data)
    if verify_performance:
        text_pred_score = text_nn.evaluate(test_dataset.data)
        assert text_pred_score == text_loaded_score_val,\
            f"Text-only network: Predicted score={text_pred_score}, " \
            f"Loaded score={text_loaded_score_val}, " \
            f"Dataset={dataset}"
    os.makedirs(f'embeddings/{dataset}/tuned_text_embedding', exist_ok=True)
    np.save(os.path.join(f'embeddings/{dataset}/tuned_text_embedding', 'train.npy'), train_text_embeddings)
    np.save(os.path.join(f'embeddings/{dataset}/tuned_text_embedding', 'test.npy'), test_text_embeddings)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset is None:
        for dataset in dataset_l:
            assert dataset in dataset_registry.list_keys()
            if args.extract_pretrained:
                extract_pretrained_embedding(dataset)
            else:
                stat_df = pd.read_csv(args.stat_df)
                extract_finetuned_embedding(dataset, stat_df,
                                            verify_performance=args.verify_performance)
    else:
        if args.use_file_name:
            dataset_to_use = []
            with open(args.dataset, 'r') as in_f:
                for line in in_f:
                    dataset_to_use.append(line.strip())
        else:
            dataset_to_use = [args.dataset]
        for dataset in dataset_to_use:
            assert dataset in dataset_registry.list_keys(), f'dataset={dataset}'
            if args.extract_pretrained:
                extract_pretrained_embedding(dataset)
            else:
                stat_df = pd.read_csv(args.stat_df)
                extract_finetuned_embedding(dataset, stat_df,
                                            verify_performance=args.verify_performance)
