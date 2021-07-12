import matplotlib.pyplot as plt
import numpy as np
import autogluon
import os
import pandas as pd
import random
import json
import time
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
from auto_mm_bench.datasets import dataset_registry, _TEXT, _NUMERICAL, _CATEGORICAL
from auto_mm_bench.utils import logging_config
import copy

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))


def update_5best(cfg):
    new_cfg = copy.deepcopy(cfg)
    new_cfg['models']['MultimodalTextModel']['search_space']['optimization.nbest'] = 5
    return new_cfg


def set_epoch3(cfg):
    new_cfg = copy.deepcopy(cfg)
    new_cfg['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = 3
    return new_cfg



@ag_text_presets.register()
def albert_base_no_hpo() -> dict:
    """The default search space that use ALBERT Base as the backbone."""
    cfg = no_hpo()
    cfg['models']['MultimodalTextModel']['search_space']['model.backbone.name'] \
        = 'google_albert_base_v2'
    cfg['models']['MultimodalTextModel']['search_space']['optimization.per_device_batch_size'] = 8
    cfg['models']['MultimodalTextModel']['search_space']['optimization.layerwise_lr_decay'] = 1.0
    return cfg


@ag_text_presets.register()
def albert_large_no_hpo() -> dict:
    """The default search space that use ALBERT Large as the backbone."""
    cfg = albert_base_no_hpo()
    cfg['models']['MultimodalTextModel']['search_space']['model.backbone.name'] \
        = 'google_albert_large_v2'
    cfg['models']['MultimodalTextModel']['search_space'][
        'optimization.per_device_batch_size'] = 4
    return cfg


@ag_text_presets.register()
def electra_small_grid_search() -> dict:
    cfg = update_5best(electra_small_no_hpo())
    cfg['hpo_params']['num_trials'] = 12
    cfg['hpo_params']['search_strategy'] = 'random'
    cfg['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = space.Categorical(5, 10)
    cfg['models']['MultimodalTextModel']['search_space']['optimization.lr'] = space.Categorical(1E-4, 5E-5)
    cfg['models']['MultimodalTextModel']['search_space']['optimization.layerwise_lr_decay'] = space.Categorical(0.8, 0.9, 1.0)
    cfg['models']['MultimodalTextModel']['search_space']['optimization.wd'] = space.Categorical(0.01, 1E-4, 0.0)
    return cfg


@ag_text_presets.register()
def electra_base_grid_search() -> dict:
    cfg = update_5best(electra_base_no_hpo())
    cfg['hpo_params']['num_trials'] = 12
    cfg['hpo_params']['search_strategy'] = 'random'
    cfg['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = space.Categorical(5, 10)
    cfg['models']['MultimodalTextModel']['search_space']['optimization.lr'] = space.Categorical(1E-4, 5E-5)
    cfg['models']['MultimodalTextModel']['search_space']['optimization.layerwise_lr_decay'] = space.Categorical(0.8, 0.9, 1.0)
    cfg['models']['MultimodalTextModel']['search_space']['optimization.wd'] = space.Categorical(0.01, 1E-4, 0.0)
    return cfg


@ag_text_presets.register()
def albert_base_grid_search() -> dict:
    cfg = update_5best(albert_base_no_hpo())
    cfg['hpo_params']['num_trials'] = 12
    cfg['hpo_params']['search_strategy'] = 'random'
    cfg['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = space.Categorical(5, 10)
    cfg['models']['MultimodalTextModel']['search_space']['model.optimization.lr'] = space.Categorical(1E-4, 5E-5),
    cfg['models']['MultimodalTextModel']['search_space']['model.optimization.wd'] = space.Categorical(0.01, 1E-4, 0.0),
    return cfg


@ag_text_presets.register()
def albert_base_all_text_epoch10():
    cfg = update_5best(albert_base_no_hpo())
    cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'concat'
    cfg['models']['MultimodalTextModel']['search_space']['preprocessing.categorical.convert_to_text'] = True
    cfg['models']['MultimodalTextModel']['search_space']['preprocessing.numerical.convert_to_text'] = True
    cfg['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = 10
    cfg['models']['MultimodalTextModel']['search_space']['optimization.layerwise_lr_decay'] = 0.8
    return cfg


@ag_text_presets.register()
def roberta_base_all_text_epoch10():
    cfg = update_5best(roberta_base_no_hpo())
    cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'concat'
    cfg['models']['MultimodalTextModel']['search_space']['preprocessing.categorical.convert_to_text'] = True
    cfg['models']['MultimodalTextModel']['search_space']['preprocessing.numerical.convert_to_text'] = True
    cfg['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = 10
    cfg['models']['MultimodalTextModel']['search_space']['optimization.layerwise_lr_decay'] = 0.8
    return cfg


@ag_text_presets.register()
def roberta_base_all_text_epoch5():
    cfg = update_5best(roberta_base_no_hpo())
    cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'concat'
    cfg['models']['MultimodalTextModel']['search_space']['preprocessing.categorical.convert_to_text'] = True
    cfg['models']['MultimodalTextModel']['search_space']['preprocessing.numerical.convert_to_text'] = True
    cfg['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = 5
    cfg['models']['MultimodalTextModel']['search_space']['optimization.layerwise_lr_decay'] = 0.8
    return cfg


@ag_text_presets.register()
def roberta_base_all_text_no_lr_decay_epoch10():
    cfg = update_5best(roberta_base_no_hpo())
    cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'concat'
    cfg['models']['MultimodalTextModel']['search_space']['preprocessing.categorical.convert_to_text'] = True
    cfg['models']['MultimodalTextModel']['search_space']['preprocessing.numerical.convert_to_text'] = True
    cfg['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = 10
    cfg['models']['MultimodalTextModel']['search_space']['optimization.layerwise_lr_decay'] = 1.0
    return cfg


@ag_text_presets.register()
def roberta_base_all_text_no_lr_decay_epoch5():
    cfg = update_5best(roberta_base_no_hpo())
    cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'concat'
    cfg['models']['MultimodalTextModel']['search_space']['preprocessing.categorical.convert_to_text'] = True
    cfg['models']['MultimodalTextModel']['search_space']['preprocessing.numerical.convert_to_text'] = True
    cfg['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = 5
    cfg['models']['MultimodalTextModel']['search_space']['optimization.layerwise_lr_decay'] = 1.0
    return cfg


@ag_text_presets.register()
def roberta_base_all_text_epoch10_stochastic_infer0():
    cfg = roberta_base_all_text_epoch10()
    cfg['models']['MultimodalTextModel']['search_space']['model.train_stochastic_chunk'] = True
    cfg['models']['MultimodalTextModel']['search_space']['model.test_stochastic_chunk'] = False
    return cfg


@ag_text_presets.register()
def roberta_base_all_text_epoch10_stochastic_infer1_repeat3():
    cfg = roberta_base_all_text_epoch10()
    cfg['models']['MultimodalTextModel']['search_space']['model.train_stochastic_chunk'] = True
    cfg['models']['MultimodalTextModel']['search_space']['model.test_stochastic_chunk'] = True
    cfg['models']['MultimodalTextModel']['search_space']['model.inference_num_repeat'] = 3
    return cfg


@ag_text_presets.register()
def roberta_base_all_text_epoch10_average3():
    cfg = roberta_base_all_text_epoch10()
    cfg['models']['MultimodalTextModel']['search_space']['model.use_avg_nbest'] = True
    cfg['models']['MultimodalTextModel']['search_space']['optimization.nbest'] = 3
    return cfg

@ag_text_presets.register()
def roberta_base_early_fusion_epoch10():
    cfg = update_5best(roberta_base_no_hpo())
    cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'attention_token'
    cfg['models']['MultimodalTextModel']['search_space']['model.network.aggregate_categorical'] = False
    return cfg


@ag_text_presets.register()
def electra_base_search_aggregator() -> dict:
    cfg = update_5best(electra_base_no_hpo())
    cfg['hpo_params']['num_trials'] = 10
    cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type']\
        = space.Categorical('attention', 'mean', 'max', 'concat')
    cfg['models']['MultimodalTextModel']['search_space']['model.network.base_feature_units'] \
        = space.Categorical(-1, 128)
    return cfg


def electra_models_with_fusion_strategies(model_type, fusion_strategy, num_epochs, average3):
    if model_type == 'small':
        cfg = update_5best(electra_small_no_hpo())
    elif model_type == 'base':
        cfg = update_5best(electra_base_no_hpo())
    elif model_type == 'large':
        cfg = update_5best(electra_large_no_hpo())
    else:
        raise NotImplementedError
    if average3:
        cfg['models']['MultimodalTextModel']['search_space']['model.use_avg_nbest'] = True
        cfg['models']['MultimodalTextModel']['search_space']['optimization.nbest'] = 3
    cfg['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = num_epochs
    if fusion_strategy == 'late_fusion_mean':
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'mean'
        cfg['models']['MultimodalTextModel']['search_space']['model.network.aggregate_categorical'] = True
    elif fusion_strategy == 'late_fusion_max':
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'max'
        cfg['models']['MultimodalTextModel']['search_space']['model.network.aggregate_categorical'] = True
    elif fusion_strategy == 'late_fusion_concat':
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'concat'
        cfg['models']['MultimodalTextModel']['search_space']['model.network.aggregate_categorical'] = True
    elif fusion_strategy == 'late_fusion_concat_gates':
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'concat'
        cfg['models']['MultimodalTextModel']['search_space']['model.network.aggregate_categorical'] = True
        cfg['models']['MultimodalTextModel']['search_space']['model.network.numerical_net.gated_activation'] = True
        cfg['models']['MultimodalTextModel']['search_space']['model.network.categorical_agg.gated_activation'] = True
    elif fusion_strategy == 'early_fusion':
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'attention_token'
        cfg['models']['MultimodalTextModel']['search_space']['model.network.aggregate_categorical'] = False
    elif fusion_strategy == 'early_fusion_layer3_units128':
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'attention_token'
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.attention_net.num_layers'] = 3
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.attention_net.units'] = 128
        cfg['models']['MultimodalTextModel']['search_space']['model.network.aggregate_categorical'] = False
    elif fusion_strategy == 'early_fusion_layer3':
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'attention_token'
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.attention_net.num_layers'] = 3
        cfg['models']['MultimodalTextModel']['search_space']['model.network.aggregate_categorical'] = False
    elif fusion_strategy == 'early_fusion_layer3_leaky':
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'attention_token'
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.attention_net.num_layers'] = 3
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.attention_net.activation'] = 'leaky'
        cfg['models']['MultimodalTextModel']['search_space']['model.network.aggregate_categorical'] = False
    elif fusion_strategy == 'early_fusion_layer3_leaky_units128':
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'attention_token'
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.attention_net.num_layers'] = 3
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.attention_net.activation'] = 'leaky'
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.attention_net.units'] = 128
        cfg['models']['MultimodalTextModel']['search_space']['model.network.aggregate_categorical'] = False
    elif fusion_strategy == 'all_text':
        cfg['models']['MultimodalTextModel']['search_space']['model.network.agg_net.agg_type'] = 'concat'
        cfg['models']['MultimodalTextModel']['search_space']['preprocessing.categorical.convert_to_text'] = True
        cfg['models']['MultimodalTextModel']['search_space']['preprocessing.numerical.convert_to_text'] = True
    else:
        raise NotImplementedError
    return cfg


for model_type in ['small', 'base', 'large']:
    for fusion_strategy in ['late_fusion_mean', 'late_fusion_max', 'late_fusion_concat',
                            'late_fusion_concat_gates',
                            'early_fusion', 'early_fusion_layer3', 'early_fusion_layer3_units128',
                            'early_fusion_layer3_leaky', 'all_text']:
        for average3 in [False, True]:
            for num_epochs in [5, 10]:
                if not average3:
                    ag_text_presets.register(f'electra_{model_type}_{fusion_strategy}_e{num_epochs}',
                                             functools.partial(electra_models_with_fusion_strategies,
                                                               model_type=model_type,
                                                               fusion_strategy=fusion_strategy,
                                                               num_epochs=num_epochs,
                                                               average3=False))
                else:
                    ag_text_presets.register(
                        f'electra_{model_type}_{fusion_strategy}_e{num_epochs}_avg3',
                        functools.partial(electra_models_with_fusion_strategies,
                                          model_type=model_type,
                                          fusion_strategy=fusion_strategy,
                                          num_epochs=num_epochs,
                                          average3=True))


@ag_text_presets.register()
def electra_base_all_text_e5_no_decay():
    cfg = ag_text_presets.create('electra_base_all_text_e5')
    cfg['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = 5
    cfg['models']['MultimodalTextModel']['search_space']['optimization.layerwise_lr_decay'] = 1.0
    return cfg


@ag_text_presets.register()
def electra_base_all_text_e10_no_decay():
    cfg = ag_text_presets.create('electra_base_all_text_e10')
    cfg['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = 10
    cfg['models']['MultimodalTextModel']['search_space']['optimization.layerwise_lr_decay'] = 1.0
    return cfg


def set_seed(seed):
    import mxnet as mx
    import torch as th
    th.manual_seed(seed)
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_parser():
    parser = argparse.ArgumentParser(description='Run AutoGluon Multimodal Tabular Benchmark.')
    parser.add_argument('--dataset', type=str,
                        choices=dataset_registry.list_keys(),
                        required=True)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--text_presets', type=str,
                        choices=ag_text_presets.list_keys() + ['no'],
                        default=None)
    parser.add_argument('--tabular_presets', type=str,
                        choices=['5fold_1stack', '3fold_1stack', 'best_quality', 'no'],
                        default=None)
    parser.add_argument('--model',
                        choices=['ag_tabular_quick',
                                 'ag_tabular_without_text',
                                 'ag_tabular_old',
                                 'ag_text_only',
                                 'ag_text_multimodal',
                                 'tabular_multimodal',       # Fuse the multimodal model to Tabular
                                 'tabular_multimodal_just_table',   # Fuse the tabular features
                                 'pre_embedding',
                                 'tune_embedding_multimodal',
                                 'tune_embedding_text'])
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--num_gpus', default=None, type=int)
    parser.add_argument('--num_folds', type=int, default=None)
    parser.add_argument('--stack_level', default=None)
    parser.add_argument('--competition',
                        action='store_true',
                        help='Use this flag to fit models on the full train+test dataset. '
                             'This can be useful for subsequent submission to a prediction competition.')
    return parser


def get_multimodal_tabular_hparam_just_gbm(text_presets):
    ret = {
        #'NN': {},
        'GBM': [
            {},
            #{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
        ],
        #'CAT': {},
        #'XGB': {},
        'TEXT_NN_V1': [text_presets],
        #'custom': ['GBM'],
    }
    return ret


def multimodal_tabular_just_table_hparam(text_presets):
    ret = {
        'NN': {},
        'GBM': [
            {},
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
        ],
        'CAT': {},
        #'XGB': {},
        'TEXT_NN_V1': [text_presets],
        'custom': ['GBM'],
    }
    return ret


TABULAR_EXCLUDE_MODELS = ['KNN', 'XT', 'RF', 'FASTAI']


def train_model(dataset_name, text_presets, save_dir,
                model,
                tabular_presets,
                num_gpus=None,
                get_competition_results=False,
                seed=123):
    set_seed(seed)
    if get_competition_results:
        train_dataset = dataset_registry.create(dataset_name, 'train')
        test_dataset = dataset_registry.create(dataset_name, 'competition')
    else:
        train_dataset = dataset_registry.create(dataset_name, 'train')
        test_dataset = dataset_registry.create(dataset_name, 'test')
    feature_columns = train_dataset.feature_columns
    label_columns = train_dataset.label_columns
    metric = train_dataset.metric
    problem_type = train_dataset.problem_type
    train_data1, tuning_data1 = sklearn.model_selection.train_test_split(
        train_dataset.data,
        test_size=0.05,
        random_state=np.random.RandomState(seed))
    train_data = train_dataset.data
    test_data = test_dataset.data
    column_types, inferred_problem_type = infer_column_problem_types(train_data1,
                                                                     tuning_data1,
                                                                     label_columns=label_columns,
                                                                     problem_type=problem_type)
    train_data = train_data[feature_columns + label_columns]
    # tuning_data = tuning_data[feature_columns + label_columns]
    if not get_competition_results:
        test_data = test_data[feature_columns + label_columns]
    train_tic = time.time()
    if model == 'ag_tabular_quick':
        MAX_NGRAM = 300
        feature_generator = AutoMLPipelineFeatureGenerator(
            vectorizer=CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=MAX_NGRAM,
                                       dtype=np.uint8))
        predictor = TabularPredictor(label=label_columns[0],
                                     path=save_dir,
                                     problem_type=problem_type)
        predictor.fit(train_data, time_limit=30,
                      feature_generator=feature_generator)
    elif model == 'ag_tabular_without_text':
        no_text_feature_columns = []
        for col_name in feature_columns:
            if column_types[col_name] != _TEXT:
                no_text_feature_columns.append(col_name)
        train_data = train_data[no_text_feature_columns + label_columns]
        # tuning_data = tuning_data[no_text_feature_columns + label_columns]
        test_data = test_data[no_text_feature_columns + label_columns]
        predictor = TabularPredictor(path=save_dir,
                                     label=label_columns[0],
                                     problem_type=problem_type,
                                     eval_metric=metric)
        if tabular_presets in ['best_quality']:
            predictor.fit(train_data=train_data,
                          excluded_model_types=TABULAR_EXCLUDE_MODELS,
                          presets=tabular_presets)
        elif tabular_presets == '5fold_1stack':
            predictor.fit(train_data=train_data,
                          excluded_model_types=TABULAR_EXCLUDE_MODELS,
                          num_bag_folds=5,
                          num_stack_levels=1)
        elif tabular_presets == 'no':
            predictor.fit(train_data=train_data,
                          excluded_model_types=TABULAR_EXCLUDE_MODELS)
        else:
            raise NotImplementedError
    elif model == 'ag_tabular_old':
        predictor = TabularPredictor(path=save_dir,
                                     label=label_columns[0],
                                     problem_type=problem_type,
                                     eval_metric=metric)
        if tabular_presets == 'best_quality':
            predictor.fit(train_data=train_data,
                          presets=tabular_presets,
                          excluded_model_types=TABULAR_EXCLUDE_MODELS)
        elif tabular_presets == '5fold_1stack':
            predictor.fit(train_data=train_data,
                          num_bag_folds=5,
                          num_stack_levels=1,
                          excluded_model_types=TABULAR_EXCLUDE_MODELS)
        elif tabular_presets == 'no':
            predictor.fit(train_data=train_data,
                          excluded_model_types=TABULAR_EXCLUDE_MODELS)
        else:
            raise NotImplementedError
    elif model == 'ag_text_only':
        text_feature_columns = [col_name for col_name in feature_columns
                                if column_types[col_name] == _TEXT]
        train_data = train_data[text_feature_columns + label_columns]
        test_data = test_data[text_feature_columns + label_columns]
        predictor = TextPredictor(path=save_dir,
                                  label=label_columns[0],
                                  problem_type=problem_type,
                                  eval_metric=metric)
        hparams = ag_text_presets.create(text_presets)
        if len(train_data) > 500000:
            hparams = set_epoch3(hparams)
        predictor.fit(train_data=train_data,
                      hyperparameters=hparams,
                      num_gpus=num_gpus,
                      seed=seed)
    elif model == 'ag_text_multimodal':
        predictor = TextPredictor(path=save_dir,
                                  label=label_columns[0],
                                  problem_type=problem_type,
                                  eval_metric=metric)
        hparams = ag_text_presets.create(text_presets)
        if len(train_data) > 500000:
            hparams = set_epoch3(hparams)
        predictor.fit(train_data=train_data,
                      hyperparameters=hparams,
                      num_gpus=num_gpus,
                      seed=seed)
    elif model == 'pre_embedding' or model == 'tune_embedding_multimodal' or model == 'tune_embedding_text':
        feature_generator = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                           enable_text_ngram_features=False)
        pre_embedding_folder = os.path.join(_CURR_DIR, 'pre_computed_embeddings')
        if model == 'pre_embedding':
            train_features = np.load(os.path.join(pre_embedding_folder, dataset_name,
                                                  'pretrain_text_embedding', 'train.npy'))
            test_features = np.load(os.path.join(pre_embedding_folder, dataset_name,
                                                 'pretrain_text_embedding', 'test.npy'))
        elif model == 'tune_embedding_multimodal':
            train_features = np.load(os.path.join(pre_embedding_folder, dataset_name,
                                                  'multimodal_embedding', 'train.npy'))
            test_features = np.load(os.path.join(pre_embedding_folder, dataset_name,
                                                 'multimodal_embedding', 'test.npy'))
        elif model == 'tune_embedding_text':
            train_features = np.load(os.path.join(pre_embedding_folder, dataset_name,
                                                  'tuned_text_embedding', 'train.npy'))
            test_features = np.load(os.path.join(pre_embedding_folder, dataset_name,
                                                 'tuned_text_embedding', 'test.npy'))
        else:
            raise NotImplementedError
        train_data = train_data.join(pd.DataFrame(
            train_features, columns=[f'pre_feat{i}' for i in range(train_features.shape[1])]))
        train_data.reset_index(drop=True, inplace=True)
        test_data = test_data.join(pd.DataFrame(
            test_features, columns=[f'pre_feat{i}' for i in range(test_features.shape[1])]))
        test_data.reset_index(drop=True, inplace=True)
        predictor = TabularPredictor(path=save_dir,
                                     label=label_columns[0],
                                     problem_type=problem_type,
                                     eval_metric=metric)
        if tabular_presets == 'best_quality':
            predictor.fit(train_data=train_data,
                          presets=tabular_presets,
                          feature_generator=feature_generator,
                          excluded_model_types=TABULAR_EXCLUDE_MODELS)
        elif tabular_presets == '5fold_1stack':
            predictor.fit(train_data=train_data,
                          num_bag_folds=5,
                          num_stack_levels=1,
                          feature_generator=feature_generator,
                          excluded_model_types=TABULAR_EXCLUDE_MODELS)
        elif tabular_presets == 'no':
            predictor.fit(train_data=train_data,
                          feature_generator=feature_generator,
                          excluded_model_types=TABULAR_EXCLUDE_MODELS)
        else:
            raise NotImplementedError

    elif model == 'tabular_multimodal' or model == 'tabular_multimodal_just_table':
        if model == 'tabular_multimodal':
            MAX_NGRAM = 300
            feature_generator = AutoMLPipelineFeatureGenerator(
                vectorizer=CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=MAX_NGRAM,
                                           dtype=np.uint8),
                enable_raw_text_features=True)
            hyperparameters = get_multimodal_tabular_hparam_just_gbm(text_presets=text_presets)
        else:
            MAX_NGRAM = 300
            feature_generator = AutoMLPipelineFeatureGenerator(
                vectorizer=CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=MAX_NGRAM,
                                           dtype=np.uint8),
                enable_raw_text_features=True,
                enable_text_special_features=False,
                enable_text_ngram_features=False
            )
            hyperparameters = multimodal_tabular_just_table_hparam(text_presets=text_presets)
        predictor = TabularPredictor(path=save_dir,
                                     label=label_columns[0],
                                     problem_type=problem_type,
                                     eval_metric=metric)
        if tabular_presets == 'best_quality':
            predictor.fit(train_data=train_data,
                          presets=tabular_presets,
                          hyperparameters=hyperparameters,
                          feature_generator=feature_generator,
                          excluded_model_types=TABULAR_EXCLUDE_MODELS)
        elif tabular_presets == '5fold_1stack':
            predictor.fit(train_data=train_data,
                          num_bag_folds=5,
                          num_stack_levels=1,
                          hyperparameters=hyperparameters,
                          feature_generator=feature_generator,
                          excluded_model_types=TABULAR_EXCLUDE_MODELS)
        elif tabular_presets == '3fold_1stack':
            predictor.fit(train_data=train_data,
                          num_bag_folds=3,
                          num_stack_levels=1,
                          hyperparameters=hyperparameters,
                          feature_generator=feature_generator,
                          excluded_model_types=TABULAR_EXCLUDE_MODELS)
        elif tabular_presets == 'no':
            predictor.fit(train_data=train_data,
                          hyperparameters=hyperparameters,
                          feature_generator=feature_generator,
                          excluded_model_types=TABULAR_EXCLUDE_MODELS)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    train_toc = time.time()
    inference_tic = time.time()
    predictions = predictor.predict(test_data, as_pandas=True)
    predictor.save()
    inference_toc = time.time()
    if problem_type == MULTICLASS or problem_type == BINARY:
        prediction_prob = predictor.predict_proba(test_data, as_pandas=True)
        prediction_prob.to_csv(os.path.join(save_dir, 'test_prediction_prob.csv'))
    predictions.to_csv(os.path.join(save_dir, 'test_prediction.csv'))
    gt = test_data[label_columns[0]]
    gt.to_csv(os.path.join(save_dir, 'ground_truth.csv'))
    if not get_competition_results:
        score = predictor.evaluate(test_data)
        with open(os.path.join(save_dir, 'test_score.json'), 'w') as of:
            json.dump({metric: score}, of)
    with open(os.path.join(save_dir, 'speed_stats.json'), 'w') as of:
        json.dump({'train_time': train_toc - train_tic,
                   'inference_time': inference_toc - inference_tic,
                   'cpuinfo': cpuinfo.get_cpu_info()}, of)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    train_model(save_dir=args.save_dir,
                model=args.model,
                dataset_name=args.dataset,
                text_presets=args.text_presets,
                tabular_presets=args.tabular_presets,
                num_gpus=args.num_gpus,
                seed=args.seed,
                get_competition_results=args.competition)
