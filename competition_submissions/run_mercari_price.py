import matplotlib.pyplot as plt
import numpy as np
import autogluon
import os
import pandas as pd
import random
import json
import time
from time import gmtime, strftime
from sklearn.feature_extraction.text import CountVectorizer
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

import platform
import functools
import cpuinfo
import argparse
import sklearn
from autogluon.tabular import TabularPredictor
from autogluon.core import space
from autogluon.core.constants import MULTICLASS, BINARY, REGRESSION
from autogluon.text import TextPredictor, ag_text_presets
from autogluon.text.text_prediction.presets import electra_small_no_hpo, electra_base_no_hpo,\
    electra_large_no_hpo, no_hpo, roberta_base_no_hpo
from autogluon.text.text_prediction.infer_types import infer_column_problem_types
from auto_mm_bench.datasets import dataset_registry, _TEXT, _NUMERICAL, _CATEGORICAL, MercariPriceSuggestion
from auto_mm_bench.utils import logging_config
import copy

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))


@ag_text_presets.register()
def electra_base_late_fusion_concate_e10_avg3():
    cfg = electra_base_no_hpo()
    cfg['models']['MultimodalTextModel']['search_space']['model.use_avg_nbest'] = True
    cfg['models']['MultimodalTextModel']['search_space']['optimization.nbest'] = 3
    cfg['models']['MultimodalTextModel']['search_space'][
        'model.network.agg_net.agg_type'] = 'concat'
    cfg['models']['MultimodalTextModel']['search_space'][
        'model.network.aggregate_categorical'] = True
    return cfg


@ag_text_presets.register()
def electra_large_late_fusion_concate_e10_avg3():
    cfg = electra_large_no_hpo()
    cfg['models']['MultimodalTextModel']['search_space']['model.use_avg_nbest'] = True
    cfg['models']['MultimodalTextModel']['search_space']['optimization.nbest'] = 3
    cfg['models']['MultimodalTextModel']['search_space'][
        'model.network.agg_net.agg_type'] = 'concat'
    cfg['models']['MultimodalTextModel']['search_space'][
        'model.network.aggregate_categorical'] = True
    return cfg


def get_tabular_hparams(text_presets):
    ret = {
        'NN': {},
        'GBM': [
            {},
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
        ],
        'CAT': {},
        'XGB': {},
        'TEXT_NN_V1': [text_presets],
        'custom': ['GBM'],
    }
    return ret


parser = argparse.ArgumentParser('Run experiments on the MachineHack '
                                 'Product Sentiment Analysis Dataset.')
parser.add_argument('--model_type', choices=['base', 'large'], default='base')
parser.add_argument('--ensemble_type', choices=['weighted', 'stack', 'text_only'],
                    default='text_only')
# parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--save_dir', default='mercari_price', type=str)
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

MAX_NGRAM = 300
feature_generator = AutoMLPipelineFeatureGenerator(
    vectorizer=CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=MAX_NGRAM,
                               dtype=np.uint8),
    enable_raw_text_features=True,
    enable_text_special_features=False,
    enable_text_ngram_features=False
)
hyperparameters = get_tabular_hparams(text_presets=electra_base_late_fusion_concate_e10_avg3)

train_dataset = MercariPriceSuggestion('train')
test_dataset = MercariPriceSuggestion('test')
competition_dataset = MercariPriceSuggestion('competition')

feature_columns = train_dataset.feature_columns
label_columns = train_dataset.label_columns

train_data = train_dataset.data
test_data = test_dataset.data
concat_df = pd.concat([train_data, test_data])
concat_df.reset_index(drop=True, inplace=True)

competition_df = competition_dataset.data[feature_columns]

if args.model_type == 'base':
    tabular_hparams = get_tabular_hparams(electra_base_late_fusion_concate_e10_avg3())
elif args.model_type == 'large':
    tabular_hparams = get_tabular_hparams(electra_large_late_fusion_concate_e10_avg3())
else:
    raise NotImplementedError

time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
if args.ensemble_type == 'weighted' or args.ensemble_type == 'stack':
    predictor = TabularPredictor(
        path=os.path.join(args.save_dir, args.model_type, time_str),
        problem_type=train_dataset.problem_type,
        eval_metric=train_dataset.metric,
        label=label_columns[0])
    if args.ensemble_type == 'weighted':
        predictor.fit(concat_df[feature_columns + [label_columns[0]]],
                      feature_generator=feature_generator,
                      hyperparameters=tabular_hparams)
    else:
        predictor.fit(concat_df[feature_columns + [label_columns[0]]],
                      feature_generator=feature_generator,
                      num_bag_folds=5,
                      num_stack_levels=1,
                      hyperparameters=tabular_hparams)
    predictor.save()
else:
    predictor = TextPredictor(path=os.path.join(args.save_dir, args.model_type, time_str),
                              problem_type=train_dataset.problem_type,
                              eval_metric=train_dataset.metric,
                              label=label_columns[0])
    predictor.fit(concat_df[feature_columns + [label_columns[0]]],
                  presets='electra_base_late_fusion_concate_e10_avg3')
    predictor.save(os.path.join(args.save_dir, args.model_type, time_str, 'text_prediction'))
predictions = predictor.predict(competition_df, as_pandas=True)
predictions.to_csv(os.path.join(args.save_dir, args.model_type, time_str,
                                'pred.csv'))
