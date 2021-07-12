"""
H2O AutoML Methods:

1) h2o_automl: Regular H2O AutoML (H2O-AutoML automatically ignores text columns)
2) h2o_word2vec: H2O word2vec featurization of text + H2O AutoML
3) h2o_embedding: Our Electra featurization of text + H2O AutoML

Notes:

To run h2o_embedding, you must first download embeddings to current working directory where this script is invoked, eg:
aws s3 sync s3://automl-mm-bench/kdd2021_embedding/20210205/embeddings pre_computed_embeddings

- This baseline only works for the first label in label_columns, ie. we predict label_columns[0]
"""

import os, random, json, time, warnings, copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psutil import virtual_memory
import platform
import functools
import cpuinfo
import argparse
import sklearn

import h2o
from h2o.estimators.word2vec import H2OWord2vecEstimator
from h2o import H2OFrame
from h2o.automl import H2OAutoML

import autogluon
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.core.constants import MULTICLASS, BINARY, REGRESSION

from auto_mm_bench.datasets import dataset_registry, _TEXT, _NUMERICAL, _CATEGORICAL
from auto_mm_bench.utils import logging_config


class H2OBaseline(object):
    """ Methods to run h2o AutoML. """
    def __init__(self):
        self.class_prefix = 'cls_'
        self.class_suffix = '_cls'
        self.label_column = None
        self.label_type = None
        self.problem_type = None
        self.classes = None

    def fit(self, train_data, label_column, problem_type, eval_metric=None,
             output_directory=None, time_limit_sec = None, random_state=123):
        self.label_column = label_column
        self.label_type = train_data[label_column].dtype
        self.problem_type = problem_type
        output_directory = os.path.join(output_directory,"h2o_model/")
        if self.problem_type in [BINARY, MULTICLASS]:
            train_data = self.add_label_prefix(train_data)

        # Set h2o memory limits recommended by the authors (recommended leaving 2 GB free for the operating system):
        t0 = time.time()
        h2o.init(log_dir=output_directory)
        train = h2o.H2OFrame(train_data)
        self.column_types = train.types
        del self.column_types[label_column] # label column will not appear in test data, remove this if it appears in test data.
        if problem_type in [BINARY, MULTICLASS]:
            train[label_column] = train[label_column].asfactor() # ensure h2o knows this is not regression
        x = list(train.columns)
        if label_column not in x:
            raise ValueError("label_column must be present in training data: %s" % label_column)
        x.remove(label_column)
        # H2O settings:
        training_params = {'seed': random_state}
        if time_limit_sec is not None:
            print("Setting time limits = ", time_limit_sec)
            training_params['max_runtime_secs'] = int(time_limit_sec)
        if eval_metric is not None: # Pass in metric to fit()
            h2o_metric = self.convert_metric(eval_metric)
            if h2o_metric is not None:
                training_params['sort_metric'] = h2o_metric
                if eval_metric != 'roc_auc': # h2o authors do not recommend using AUC for early-stopping in binary classification and suggest using default instead. We also empirically verified this works better as well.
                    training_params['stopping_metric'] = h2o_metric  # TODO: Not used in AutoMLBenchmark! Do we keep this? Authors mentioned to do this in email.
            else:
                warnings.warn("Specified metric is unknown to h2o. Fitting h2o without supplied evaluation metric instead.")
        h2o_model = H2OAutoML(**training_params)
        h2o_model.train(x=x, y=label_column, training_frame=train)
        if self.problem_type in [BINARY, MULTICLASS]:
            train_data = self.remove_label_prefix(train_data)
        t1 = time.time()
        fit_time = t1 - t0
        num_models_trained = len(h2o_model.leaderboard)
        # Get num_models_ensemble:
        if not h2o_model.leader:
            raise AssertionError("H2O could not produce any model in the requested time.")

        best_model = h2o_model.leader
        h2o.save_model(best_model, path=output_directory)
        self.model = h2o_model
        return (num_models_trained, fit_time)

    def predict(self, test_data, predict_proba = False, pred_class_and_proba = False):
        """ Use pred_class_and_proba to produce both predicted probabilities and predicted classes.
            If this is regression problem, predict_proba and pred_class_and_proba are disregarded.
            Label column should not be present in test_data.

            Returns: Tuple (y_pred, y_prob, inference_time) where any element may be None.
            y_prob is a 2D numpy array of predicted probabilities, where each column represents a class. The ith column represents the class found via: self.classes[i]
        """
        h2o_model = self.model
        if self.problem_type == REGRESSION:
            pred_class_and_proba = False
            predict_proba = False
        y_pred = None
        y_prob = None
        t0 = time.time()
        test = h2o.H2OFrame(test_data, column_types=self.column_types)
        preds_df = h2o_model.predict(test).as_data_frame(use_pandas=True)
        t1 = time.time()
        predict_time = t1 - t0
        if self.problem_type is not REGRESSION:
            self.classes = preds_df.columns.tolist()[1:]
            if self.problem_type in [BINARY, MULTICLASS]:
                self.classes = self.remove_label_prefix_class(self.classes)

        if (not predict_proba) or pred_class_and_proba:
            y_pred = preds_df.iloc[:, 0]
            # print(y_pred[:5])
            if self.problem_type in [BINARY, MULTICLASS]:
                y_pred = pd.Series(self.remove_label_prefix_class(list(y_pred.values)), index=y_pred.index)
            # print(y_pred[:5])

        if predict_proba or pred_class_and_proba:
            y_prob = preds_df.iloc[:, 1:] # .values  # keep as pandas instead of numpy

        # Shutdown H2O before returning value:
        if h2o.connection():
            h2o.remove_all()
            h2o.connection().close()
        if h2o.connection().local_server:
            h2o.connection().local_server.shutdown()
        return (y_pred, y_prob, predict_time)

    def convert_metric(self, metric):
        """Converts given metric to appropriate h2o metric used for sort_metric.
           Args:
                metric : str
                    May take one of the following values:
        """
        metrics_map = {
            'accuracy': 'AUTO',
            'acc': 'AUTO',
            'f1': 'auc',
            'log_loss': 'logloss',
            'roc_auc': 'auc',
            'balanced_accuracy': 'mean_per_class_error',
            'precision': 'auc',
            'recall': 'auc',
            'mean_squared_error': 'mse',
            'root_mean_squared_error': 'mse',
            'median_absolute_error': 'mae',
            'mean_absolute_error': 'mae',
            # 'r2': 'deviance',
            'r2': 'mse',
        }
        if metric in metrics_map:
            return metrics_map[metric]
        else:
            warnings.warn("Unknown metric will not be used by h2o: %s" % metric)
            return None

    # Present to deal with defect in H2O regarding altering of class label names
    def add_label_prefix(self, df):
        # print(df[self.label_column].iloc[0])
        df = df.copy()
        df[self.label_column] = [self.class_prefix + str(label[0]) + self.class_suffix for label in zip(df[self.label_column])]
        # print(df[self.label_column].iloc[0])
        return df

    # Present to deal with defect in H2O regarding altering of class label names
    def remove_label_prefix(self, df):
        df = df.copy()
        length_to_remove_prefix = len(self.class_prefix)
        length_to_remove_suffix = len(self.class_suffix)
        # print(df[self.label_column].iloc[0])
        df[self.label_column] = [label[0][length_to_remove_prefix:] for label in zip(df[self.label_column])]
        df[self.label_column] = [label[0][:-length_to_remove_suffix] for label in zip(df[self.label_column])]
        # print(df[self.label_column].iloc[0])
        df[self.label_column] = df[self.label_column].astype(self.label_type)
        return df

    # Present to deal with defect in H2O regarding altering of class label names
    def remove_label_prefix_class(self, class_name_list):
        length_to_remove_prefix = len(self.class_prefix)
        length_to_remove_suffix = len(self.class_suffix)
        # print(class_name_list)
        class_name_list = [label[length_to_remove_prefix:] for label in class_name_list]
        class_name_list = [label[:-length_to_remove_suffix] for label in class_name_list]
        # print(class_name_list)
        class_name_list = np.array(class_name_list, dtype=self.label_type)
        class_name_list = class_name_list.tolist()
        # print(class_name_list)
        return class_name_list


# Non-class methods:
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def get_text_cols(df):
    feature_generator = AutoMLPipelineFeatureGenerator(enable_text_ngram_features=False, enable_text_special_features=False,enable_raw_text_features=True)
    feature_generator.fit(df)
    return feature_generator.feature_metadata_in.type_group_map_special['text']

def train_w2v(df, epochs=None, save_dir=None):
    """ trains word2vec model on all text columns of df.
        Returns w2v model object that can transform data.
    """
    print("training word2vec model ...")
    args = {}
    if epochs is not None:
        args['epochs'] = int(epochs)
    if save_dir is not None:
        args['export_checkpoints_dir'] = os.path.join(save_dir,"h2o_model/")

    df = df.copy()
    text_columns = get_text_cols(df)
    print("Text columns are: ", text_columns)
    df_text = df[text_columns]
    text_frame = H2OFrame(df_text)
    for col in text_columns:
        text_frame[col] = text_frame[col].ascharacter()

    words = text_frame.tokenize(" ")
    w2v_model = H2OWord2vecEstimator(sent_sample_rate = 0.0, **args)
    w2v_model.train(training_frame=words)
    w2v_model.text_columns = text_columns
    return w2v_model

def process_w2v(df, w2v_model):
    """ returns new df with text-features all replaced by word2vec features """
    print("processind data with word2vec ...")
    df = df.copy()
    text_columns = w2v_model.text_columns
    df_text = df[text_columns]
    text_frame = H2OFrame(df_text)
    for col in text_columns:
        text_frame[col] = text_frame[col].ascharacter()

    words = text_frame.tokenize(" ")
    text_feats = w2v_model.transform(words, aggregate_method = "AVERAGE")
    text_feats = text_feats.as_data_frame()
    df.drop(columns=text_columns, inplace=True)
    return pd.concat([df,text_feats], axis=1).reset_index()

def get_embedded(train_data, test_data, dataset_name, embed_dir=None):
    """ Returns version of DFs with text fields embedded by pretrained Electra.
        Only consider pretrained embeddings for H2O
        embed_dir = path to where embedding files are.
    """
    print("fetching embeddings ...")
    if embed_dir is None: # search in current directory for embedding files
        embed_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))

    pre_embedding_folder = os.path.join(embed_dir, 'pre_computed_embeddings')
    train_features = np.load(os.path.join(pre_embedding_folder, dataset_name,
                                          'pretrain_text_embedding', 'train.npy'))
    test_features = np.load(os.path.join(pre_embedding_folder, dataset_name,
                                         'pretrain_text_embedding', 'test.npy'))
    text_feature_names = get_text_cols(train_data)
    train_data = train_data.copy()
    train_data.drop(columns=text_feature_names, inplace=True)
    train_data = train_data.join(pd.DataFrame(
        train_features, columns=[f'pre_feat{i}' for i in range(train_features.shape[1])]))
    train_data.reset_index(drop=True, inplace=True)
    test_data = test_data.copy()
    test_data.drop(columns=text_feature_names, inplace=True)
    test_data = test_data.join(pd.DataFrame(
        test_features, columns=[f'pre_feat{i}' for i in range(test_features.shape[1])]))
    test_data.reset_index(drop=True, inplace=True)
    print("text has been replaced with embeddings. Dimensions of post-embedding train & test data:")
    print(train_data.shape)
    print(test_data.shape)
    return train_data, test_data


def get_parser():
    parser = argparse.ArgumentParser(description='Run AutoGluon Multimodal Tabular Benchmark.')
    parser.add_argument('--dataset', type=str,
                        choices=dataset_registry.list_keys(),
                        required=True)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--baseline',
                        choices=['h2o_automl',
                                 'h2o_word2vec',
                                 'h2o_embedding',
                                 ])
    parser.add_argument('--time_limit_sec', type=str,
                        choices=[str(x) for x in [60, 120, 300, 600, 3600, 10800, 14400, 28800]],
                        default=None)
    parser.add_argument('--w2v_epochs', default=None, type=int)
    parser.add_argument('--embed_dir', default=None, type=str)
    parser.add_argument('--seed', default=123, type=int)

    return parser

def train_baseline(dataset_name, save_dir, baseline,
                   time_limit_sec=None, w2v_epochs=None, embed_dir=None, seed=123):
    set_seed(seed)
    train_dataset = dataset_registry.create(dataset_name, 'train')
    test_dataset = dataset_registry.create(dataset_name, 'test')
    feature_columns = train_dataset.feature_columns
    label_columns = train_dataset.label_columns
    label_column = label_columns[0]
    eval_metric = train_dataset.metric
    problem_type = train_dataset.problem_type
    train_data = train_dataset.data
    test_data = test_dataset.data
    train_data = train_data[feature_columns + label_columns]
    test_data = test_data[feature_columns + label_columns]
    print("Running: ",baseline)

    # Train baseline:
    train_tic = time.time()
    if baseline == 'h2o_word2vec':
        h2o.init()
        w2v_model = train_w2v(train_data, epochs=w2v_epochs, save_dir=save_dir)
        train_data = process_w2v(train_data, w2v_model)
        test_data = process_w2v(test_data, w2v_model)
    elif baseline == 'h2o_embedding':
        train_data, test_data = get_embedded(train_data, test_data, dataset_name, embed_dir=embed_dir)

    print("Train/test data shapes: ")
    print(train_data.shape)
    print(test_data.shape)
    h2o_model = H2OBaseline()
    num_models_trained, fit_time = h2o_model.fit(train_data=train_data,
                label_column=label_column, problem_type=problem_type, eval_metric=eval_metric,
                time_limit_sec=time_limit_sec, output_directory=save_dir)
    train_toc = time.time()
    print("H2O fit runtime: %s" % fit_time)

    # Predict with baseline:
    inference_tic = time.time()
    y_pred, y_prob, predict_time = h2o_model.predict(test_data.drop(columns=[label_column]),
                                                     pred_class_and_proba=True)
    inference_toc = time.time()
    print("H2O predict runtime: %s" % predict_time)

    # Evaluate predictions:
    # class_order = h2o_model.classes
    preds_toevaluate = y_pred
    if eval_metric is not None:
        if eval_metric == 'roc_auc':
            preds_toevaluate = y_prob.iloc[:,1]
        elif eval_metric == 'log_loss':
            preds_toevaluate = y_prob

    gt = test_data[label_column]
    gt.to_csv(os.path.join(save_dir, 'ground_truth.csv'))
    y_pred.to_csv(os.path.join(save_dir, 'h2o_test_prediction.csv'))
    if problem_type == MULTICLASS or problem_type == BINARY:
        y_prob.to_csv(os.path.join(save_dir, 'h2o_test_prediction_prob.csv'))
    if len(gt) != len(y_pred):
        print("WARNING: length of gt, y_pred dont match!")
        print("len(gt) ",len(gt))
        print("len(y_pred) ",len(y_pred))
        print("test_data.shape ", test_data.shape)
        if len(y_pred) > len(gt):
            print("WARNING: truncating predictions length to length of labels in test data ...")
            y_pred = y_pred[:len(gt)]
            y_prob = y_prob[:len(gt)]

    scorer = TabularPredictor(label=label_column, problem_type=problem_type, eval_metric=eval_metric)
    # scorer.fit(train_data.sample(:200], hyperparameters={'GBM': {'num_boost_round': 1}}, presets='ignore_text')
    score = scorer._learner.eval_metric(gt, preds_toevaluate)
    print("H2O score: ", score)
    with open(os.path.join(save_dir, 'test_score.json'), 'w') as of:
        json.dump({eval_metric: score}, of)
    with open(os.path.join(save_dir, 'speed_stats.json'), 'w') as of:
        json.dump({'train_time': train_toc - train_tic,
                   'inference_time': inference_toc - inference_tic,
                   'cpuinfo': cpuinfo.get_cpu_info()}, of)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    train_baseline(dataset_name=args.dataset,
                   save_dir=args.save_dir,
                   baseline=args.baseline,
                   time_limit_sec=args.time_limit_sec,
                   w2v_epochs=args.w2v_epochs,
                   embed_dir=args.embed_dir,
                   seed=args.seed)
