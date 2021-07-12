from autogluon.tabular import TabularPredictor
from autogluon.text.text_prediction.mx.models import MultiModalTextModel
from autogluon.core.metrics import get_metric
import os
import pandas as pd
import boto3
from auto_mm_bench.datasets import MercariPriceSuggestion100K, dataset_registry
from autogluon.core.utils import compute_permutation_feature_importance


MULTIMODAL_TEXT_MODEL_NAME = 'ag_text_multimodal-electra_base_late_fusion_concat_e10_avg3-no'
STACK_ENSEMBLE_MODEL_NAME = 'tabular_multimodal_just_table-electra_base_late_fusion_concat_e10_avg3-5fold_1stack'
TABULAR_MODEL_NAME = 'ag_tabular_old-no-5fold_1stack'


download_path = 'download_model_path'

dataset_l = [
    "product_sentiment_machine_hack",
    "google_qa_question_type_reason_explanation",
    "google_qa_answer_type_reason_explanation",
    "women_clothing_review",
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

stat_df = pd.read_csv('auto_mm_benchmark_result/path_123.csv', index_col=0)


def estimate_importance(dataset, model_name):
    if os.path.exists(os.path.join('feature_importance', dataset,
                                   model_name, 'importance.csv')):
        print(f'Found {dataset}, {model_name}')
        return
    model_remote_path = stat_df.loc[model_name, dataset]
    postfix = '/test_score.json'

    remote_dir_name = model_remote_path[:-len(postfix)]

    def downloadDirectoryFroms3(bucketName, remoteDirectoryName, local_dir_path):
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(bucketName)
        for obj in bucket.objects.filter(Prefix=remoteDirectoryName):
            print(obj.key)
            download_path = os.path.join(local_dir_path, obj.key)
            if not os.path.exists(os.path.dirname(download_path)):
                os.makedirs(os.path.dirname(download_path), exist_ok=True)
            bucket.download_file(obj.key, download_path)
    local_dir_name = os.path.join(download_path, remote_dir_name)
    if os.path.exists(local_dir_name):
        pass
    else:
        downloadDirectoryFroms3('automl-mm-bench', remote_dir_name, download_path)
    test_dataset = dataset_registry.create(dataset, 'test')
    if model_name == MULTIMODAL_TEXT_MODEL_NAME:
        predictor = MultiModalTextModel.load(os.path.join(local_dir_name, 'saved_model'))
    elif model_name == TABULAR_MODEL_NAME:
        predictor = TabularPredictor.load(os.path.join(local_dir_name))
    elif model_name == STACK_ENSEMBLE_MODEL_NAME:
        predictor = TabularPredictor.load(os.path.join(local_dir_name))
    else:
        raise NotImplementedError
    sample_size = min(len(test_dataset.data), 1000)
    if model_name == TABULAR_MODEL_NAME:
        importance_df = predictor.feature_importance(test_dataset.data[test_dataset.feature_columns + test_dataset.label_columns],
                                                     subsample_size=sample_size)
    else:
        importance_df = compute_permutation_feature_importance(
            test_dataset.data[test_dataset.feature_columns],
            test_dataset.data[test_dataset.label_columns[0]],
            predict_func=predictor.predict,
            eval_metric=get_metric(test_dataset.metric),
            subsample_size=sample_size,
            num_shuffle_sets=3
        )
    os.makedirs(os.path.join('feature_importance', dataset, model_name), exist_ok=True)
    importance_df.to_csv(os.path.join('feature_importance', dataset,
                                      model_name, 'importance.csv'))
    print(importance_df)


for dataset in dataset_l:
    estimate_importance(dataset, TABULAR_MODEL_NAME)

for dataset in dataset_l:
    estimate_importance(dataset, MULTIMODAL_TEXT_MODEL_NAME)

for dataset in dataset_l:
    estimate_importance(dataset, STACK_ENSEMBLE_MODEL_NAME)
