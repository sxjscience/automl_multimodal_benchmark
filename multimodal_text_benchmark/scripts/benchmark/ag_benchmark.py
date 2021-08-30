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
from autogluon.text.text_prediction.infer_types import infer_column_problem_types
from auto_mm_bench.datasets import dataset_registry, _TEXT, _NUMERICAL, _CATEGORICAL, TEXT_BENCHMARK_ALIAS_MAPPING
from auto_mm_bench.utils import logging_config
import copy

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))



def set_seed(seed):
    import mxnet as mx
    import torch as th
    th.manual_seed(seed)
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_parser():
    parser = argparse.ArgumentParser(description='Run AutoGluon Multimodal Text+Tabular Benchmark.')
    parser.add_argument('--nickname', type=str,
                        choices=sorted(list(TEXT_BENCHMARK_ALIAS_MAPPING.keys())),
                        default=None)
    parser.add_argument('--dataset', type=str,
                        choices=dataset_registry.list_keys(),
                        default=None)
    parser.add_argument('--embedding_dir', type=str,
                        help='Directory of the features extracted from a fine-tuned neural network',
                        default=None)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--text_backbone',
                        choices=['roberta_base', 'electra_base'], default='electra_base')
    parser.add_argument('--multimodal_fusion_strategy',
                        choices=['fuse_late', 'fuse_early', 'all_text'], default='fuse_late')
    parser.add_argument('--decay-rate', default=0.8, type=float)
    parser.add_argument('--n-average-epoch', default=3, type=int)
    parser.add_argument('--ensemble_option', type=str, choices=['5fold_1stack', 'weighted'], default='weighted')
    parser.add_argument('--model',
                        choices=['ag_tabular_without_text',           # AG Tabular model without the text column
                                 'ag_tabular_without_multimodal_nn',  # AG Tabular model without multimodal text nn
                                 'ag_text_only',                      # AG Text model (multimodal text nn) with only text and no tabular features.
                                 'ag_text_multimodal',                # AG Text model (multimodal text nn) with text + tabular features.
                                 'ag_tabular_multimodal',             # AG Tabular model with the multimodal text nn fused.
                                 'pre_embedding',                     # Use the pre_embedding (embedding extracted without finetuning the text model)
                                 'tune_embedding',                    # Use the embedding that is obtained by finetuning the text-only/multimodal-text-tabular model
                                 ],
                        required=True)
    parser.add_argument('--extract_embedding', action='store_true',
                        help='Whether to extract the embedding of the training set at the end of training.')
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--competition',
                        action='store_true',
                        help='Train on the full dataset. '
                             'This can be used for submission.')
    return parser


def set_n_average_epoch(cfg, nbest_epoch=3):
    """Set the number of epochs to be 3 to reduce time for large datasets."""
    new_cfg = copy.deepcopy(cfg)
    new_cfg['models']['MultimodalTextModel']['search_space']['optimization.nbest'] = nbest_epoch
    return new_cfg


def set_lr_decay(cfg, decay_rate):
    new_cfg = copy.deepcopy(cfg)
    new_cfg['models']['MultimodalTextModel']['search_space']['optimization.layerwise_lr_decay'] = decay_rate
    return new_cfg


def register_text_config(text_backbone, multimodal_fusion_strategy, decay_rate, n_average_epoch):
    base_key = f'{text_backbone}_{multimodal_fusion_strategy}'
    cfg = ag_text_presets.create(base_key)
    cfg = set_lr_decay(cfg, decay_rate)
    cfg = set_n_average_epoch(cfg, n_average_epoch)

    new_key = f'{text_backbone}_{multimodal_fusion_strategy}_decay{decay_rate}_avg{n_average_epoch}'

    def foo():
        return cfg

    ag_text_presets.register(new_key, foo)
    return new_key, cfg


def disable_text_training(cfg):
    new_cfg = copy.deepcopy(cfg)
    new_cfg['models']['MultimodalTextModel']['search_space'][
        'model.num_trainable_layers'] = 0
    new_cfg['models']['MultimodalTextModel']['search_space'][
        'model._disable_update'] = True
    new_cfg['models']['MultimodalTextModel']['search_space'][
        'optimization.num_train_epochs'] = 1
    new_cfg['models']['MultimodalTextModel']['search_space'][
        'preprocessing.categorical.convert_to_text'] = True
    new_cfg['models']['MultimodalTextModel']['search_space']['optimization.lr'] = 0.
    return new_cfg


def get_ag_tabular_hparam(text_presets, use_multimodal_text_nn=True):
    ret = {
        'NN': {},
        'GBM': [
            {},
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
            'GBMLarge',
        ],
        'CAT': {},
        'XGB': {},
    }
    if use_multimodal_text_nn:
        ret['AG_TEXT_NN'] = [text_presets]
    return ret


def train_model(args):
    seed = args.seed
    dataset_name = args.dataset
    save_dir = args.save_dir
    get_competition_results = args.competition
    model = args.model
    if save_dir is None:
        save_dir = f"{dataset_name}_{model}_{args.text_backbone}_{args.decay_rate}_{args.n_average_epoch}_{args.ensemble_option}_{args.multimodal_fusion_strategy}"

    set_seed(seed)
    if get_competition_results:
        train_dataset = dataset_registry.create(dataset_name, 'train')
        dev_dataset = dataset_registry.create(dataset_name, 'test')
        test_dataset = dataset_registry.create(dataset_name, 'competition')
        train_data = pd.concat([train_dataset.data, dev_dataset.data])
        test_data = test_dataset.data
    else:
        train_dataset = dataset_registry.create(dataset_name, 'train')
        test_dataset = dataset_registry.create(dataset_name, 'test')
        train_data = train_dataset.data
        test_data = test_dataset.data

    feature_columns = train_dataset.feature_columns
    label_columns = train_dataset.label_columns
    metric = train_dataset.metric
    problem_type = train_dataset.problem_type
    train_data1, tuning_data1 = sklearn.model_selection.train_test_split(
        train_data,
        test_size=0.05,
        random_state=np.random.RandomState(seed))
    column_types, inferred_problem_type = infer_column_problem_types(train_data1,
                                                                     tuning_data1,
                                                                     label_columns=label_columns,
                                                                     problem_type=problem_type)
    train_data = train_data[feature_columns + label_columns]

    if not get_competition_results:
        test_data = test_data[feature_columns + label_columns]

    text_presets_key, text_presets = register_text_config(text_backbone=args.text_backbone,
                                                          multimodal_fusion_strategy=args.multimodal_fusion_strategy,
                                                          decay_rate=args.decay_rate,
                                                          n_average_epoch=args.n_average_epoch)

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
    elif model == 'ag_tabular_without_text' or model == 'ag_tabular_without_multimodal_nn' or model == 'ag_tabular_multimodal':
        if model == 'ag_tabular_without_text':
            no_text_feature_columns = []
            for col_name in feature_columns:
                if column_types[col_name] != _TEXT:
                    no_text_feature_columns.append(col_name)
            train_data = train_data[no_text_feature_columns + label_columns]
            test_data = test_data[no_text_feature_columns + label_columns]
        if model == 'ag_tabular_multimodal':
            hyperparameters = get_ag_tabular_hparam(text_presets=text_presets,
                                                    use_multimodal_text_nn=True)
        else:
            hyperparameters = get_ag_tabular_hparam(text_presets=text_presets,
                                                    use_multimodal_text_nn=False)
        predictor = TabularPredictor(path=save_dir,
                                     label=label_columns[0],
                                     problem_type=problem_type,
                                     eval_metric=metric)

        if args.ensemble_option == '5fold_1stack':
            predictor.fit(train_data=train_data,
                          hyperparameters=hyperparameters,
                          num_bag_folds=5,
                          num_stack_levels=1)
        elif args.ensemble_option == 'weighted':
            predictor.fit(train_data=train_data,
                          hyperparameters=hyperparameters)
        else:
            raise NotImplementedError
    elif model == 'ag_text_only' or model == 'ag_text_multimodal':
        if model == 'ag_text_only':
            text_feature_columns = [col_name for col_name in feature_columns
                                    if column_types[col_name] == _TEXT]
            train_data = train_data[text_feature_columns + label_columns]
            test_data = test_data[text_feature_columns + label_columns]
        predictor = TextPredictor(path=save_dir,
                                  label=label_columns[0],
                                  problem_type=problem_type,
                                  eval_metric=metric)
        predictor.fit(train_data=train_data,
                      hyperparameters=text_presets,
                      seed=seed)
    elif model == 'pre_embedding' or model == 'tune_embedding':
        if model == 'pre_embedding':
            text_presets_no_train = disable_text_training(text_presets)
            text_feature_columns = [col_name for col_name in train_dataset.feature_columns
                                    if column_types[col_name] == 'text']
            train_text_only_data = train_dataset.data[text_feature_columns + train_dataset.label_columns]
            test_text_only_data = test_dataset.data[text_feature_columns + test_dataset.label_columns]
            sampled_train_data = train_text_only_data.sample(100)
            predictor = TextPredictor(path=os.path.join(save_dir, 'temp_extractor'),
                                      label=label_columns[0],
                                      problem_type=problem_type,
                                      eval_metric=metric)
            predictor.fit(train_data=sampled_train_data,
                          hyperparameters=text_presets_no_train,
                          seed=seed)
            train_features = predictor.extract_embedding(train_data, as_pandas=False)
            test_features = predictor.extract_embedding(test_data, as_pandas=False)
            if args.extract_embedding:
                np.save(os.path.join(save_dir, 'train_embedding.npy'), train_features)
                np.save(os.path.join(save_dir, 'test_embedding.npy'), test_features)
                return
        else:
            train_features = np.load(os.path.join(args.embedding_dir, 'train_embedding.npy'))
            test_features = np.load(os.path.join(args.embedding_dir, 'test_embedding.npy'))
        train_data = train_data.join(pd.DataFrame(
            train_features, columns=[f'pre_feat{i}' for i in range(train_features.shape[1])]))
        train_data.reset_index(drop=True, inplace=True)
        test_data = test_data.join(pd.DataFrame(
            test_features, columns=[f'pre_feat{i}' for i in range(test_features.shape[1])]))
        test_data.reset_index(drop=True, inplace=True)
        hyperparameters = get_ag_tabular_hparam(text_presets=text_presets,
                                                use_multimodal_text_nn=False)
        predictor = TabularPredictor(path=save_dir,
                                     label=label_columns[0],
                                     problem_type=problem_type,
                                     eval_metric=metric)

        if args.ensemble_option == '5fold_1stack':
            predictor.fit(train_data=train_data,
                          hyperparameters=hyperparameters,
                          num_bag_folds=5,
                          num_stack_levels=1)
        elif args.ensemble_option == 'weighted':
            predictor.fit(train_data=train_data,
                          hyperparameters=hyperparameters)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    train_toc = time.time()
    inference_tic = time.time()
    print('Begin to run inference')
    predictions = predictor.predict(test_data, as_pandas=True)

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
    print({metric: score})
    # Start to extract embedding
    if args.extract_embedding:
        print('Start to extract embeddings')
        train_embeddings = predictor.extract_embedding(train_data, as_pandas=False)
        test_embeddings = predictor.extract_embedding(test_data, as_pandas=False)
        np.save(os.path.join(save_dir, 'train_embedding.npy'), train_embeddings)
        np.save(os.path.join(save_dir, 'test_embedding.npy'), test_embeddings)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.nickname is None and args.dataset is None:
        raise NotImplementedError('Nickname and dataset are both None. This is not supported')
    if args.nickname is not None:
        args.dataset = TEXT_BENCHMARK_ALIAS_MAPPING[args.nickname]
    train_model(args)
