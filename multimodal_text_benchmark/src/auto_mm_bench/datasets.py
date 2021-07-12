import pandas as pd
import os
import abc
import numpy as np
from .utils import download
from .base import get_repo_url, get_data_home_dir
from .registry import Registry

_TEXT = 'text'
_CATEGORICAL = 'categorical'
_NUMERICAL = 'numerical'

_CLASSIFICATION = 'classification'
_BINARY = 'binary'
_MULTICLASS = 'multiclass'
_REGRESSION = 'regression'

dataset_registry = Registry('auto_mm_bench_datasets')


class BaseMultiModalDataset(abc.ABC):
    @property
    @abc.abstractmethod
    def feature_columns(self):
        pass

    @property
    @abc.abstractmethod
    def label_columns(self):
        pass

    @property
    def label_types(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def data(self):
        pass

    @property
    @abc.abstractmethod
    def metric(self):
        pass

    @property
    @abc.abstractmethod
    def problem_type(self):
        pass


@dataset_registry.register('product_sentiment_machine_hack')
class MachineHackSentimentPrediction(BaseMultiModalDataset):
    _SOURCE = 'https://www.machinehack.com/hackathons/product_sentiment_classification_weekend_hackathon_19/overview'
    _INFO = {
        'train': {
            'url': get_repo_url() + 'machine_hack_product_sentiment/train.csv',
            'sha1sum': '65f47ed9f760b096f4181d8fefadd4638a75c838'
        },
        'test': {
            'url': get_repo_url() + 'machine_hack_product_sentiment/dev.csv',
            'sha1sum': '8ef955f5ede1a7bd7ee2a0e7cea304026f2bf283'
        },
        'competition': {
            'url': get_repo_url() + 'machine_hack_product_sentiment/test.csv',
            'sha1sum': 'fc714eaeef2710990aea1c049315231c1678a061'
        }
    }

    def __init__(self, split='train'):
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  'machine_hack_sentiment_analysis',
                                  f'{split}.csv')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_csv(self._path)

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def feature_columns(self):
        return ['Product_Description', 'Product_Type']

    @property
    def feature_types(self):
        return [_TEXT, _CATEGORICAL]

    @property
    def label_columns(self):
        return ['Sentiment']

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def metric(self):
        return 'acc'

    @property
    def problem_type(self):
        return _MULTICLASS


@dataset_registry.register('jigsaw_unintended_bias')
class JigsawUnintendedBias(BaseMultiModalDataset):
    _SOURCE = 'https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification',
    _INFO = {
        'train': {
            'url': get_repo_url() + 'jigsaw_unintended_bias/train.pq',
            'sha1sum': '4215c898cd670c56d0e41b7d021b2f7527e36b42'
        },
        'test': {
            'url': get_repo_url() + 'jigsaw_unintended_bias/dev.pq',
            'sha1sum': '9851fa0cd408446e92c5b1508fd7d918358e5121'
        },
        'competition': {
            'url': get_repo_url() + 'jigsaw_unintended_bias/test.pq',
            'sha1sum': '53792e81830501b1feb404d99d25a6a3697f3a44'
        }
    }

    _LOCAL_NAME = 'jigsaw_unintended_bias'

    def __init__(self, split='train'):
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  self._LOCAL_NAME,
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_parquet(self._path)
        self._text_columns = ['comment_text']
        self._identity_attribute_columns =\
            ['asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian', 'female',
             'heterosexual', 'hindu',
             'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability', 'jewish',
             'latino', 'male', 'muslim', 'other_disability',
             'other_gender', 'other_race_or_ethnicity', 'other_religion',
             'other_sexual_orientation', 'physical_disability',
             'psychiatric_or_mental_illness', 'transgender', 'white']
        self._user_voting_columns = ['funny', 'wow', 'sad', 'likes', 'disagree']

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def feature_columns(self):
        return self._text_columns + self._identity_attribute_columns + self._user_voting_columns

    @property
    def feature_types(self):
        return [_TEXT] * len(self._text_columns)\
               + [_NUMERICAL] * (len(self._identity_attribute_columns) +
                                  len(self._user_voting_columns))

    @property
    def fill_na_values(self):
        ret = dict()
        for col in self._identity_attribute_columns:
            ret[col] = 0
        for col in self._user_voting_columns:
            ret[col] = 0
        return ret

    @property
    def label_columns(self):
        return ['target']

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def metric(self):
        return 'roc_auc'

    @property
    def problem_type(self):
        return _BINARY


@dataset_registry.register('jigsaw_unintended_bias100K')
class JigsawUnintendedBias100K(JigsawUnintendedBias):
    _SOURCE = 'https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification',
    _INFO = {
        'train': {
            'url': get_repo_url() + 'jigsaw_unintended_bias100K/train.pq',
            'sha1sum': '557d0349d6bdd616c8631d6db844855d76810010'
        },
        'test': {
            'url': get_repo_url() + 'jigsaw_unintended_bias100K/test.pq',
            'sha1sum': '5b1d41c8dd36f43f3cd9b50a921a59d422e28774'
        },
    }

    _LOCAL_NAME = 'jigsaw_unintended_bias100k'

    def __init__(self, split='train'):
        super().__init__(split)


@dataset_registry.register('google_qa_label')
class GoogleQuestQALabel(BaseMultiModalDataset):
    _SOURCE = 'https://www.kaggle.com/c/google-quest-challenge/data',
    _INFO = {
        'train': {
            'url': get_repo_url() + 'google_quest_qa/train.pq',
            'sha1sum': 'f425497492b224358ddca15d2848ea38cf76e6e3'
        },
        'test': {
            'url': get_repo_url() + 'google_quest_qa/dev.pq',
            'sha1sum': '7d1a5e679f6fef37025ac50933c664c3340bb18c'
        },
        'competition': {
            'url': get_repo_url() + 'google_quest_qa/test.pq',
            'sha1sum': '2549d1e402b00f6a91a9e001962ee0b4a71075f2'
        }
    }

    def __init__(self, split='train'):
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  'google_quest_qa',
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_parquet(self._path)

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def feature_columns(self):
        return ['question_title', 'question_body', 'answer', 'category']

    @property
    def feature_types(self):
        return [_TEXT, _TEXT, _TEXT, _CATEGORICAL]

    @property
    def label_columns(self):
        return ['question_asker_intent_understanding',
                'question_body_critical', 'question_conversational',
                'question_expect_short_answer', 'question_fact_seeking',
                'question_has_commonly_accepted_answer',
                'question_interestingness_others', 'question_interestingness_self',
                'question_multi_intent', 'question_not_really_a_question',
                'question_opinion_seeking', 'question_type_choice',
                'question_type_compare', 'question_type_consequence',
                'question_type_definition', 'question_type_entity',
                'question_type_instructions', 'question_type_procedure',
                'question_type_reason_explanation', 'question_type_spelling',
                'question_well_written', 'answer_helpful',
                'answer_level_of_information', 'answer_plausible', 'answer_relevance',
                'answer_satisfaction', 'answer_type_instructions',
                'answer_type_procedure', 'answer_type_reason_explanation',
                'answer_well_written']

    @property
    def label_types(self):
        return [_NUMERICAL] * len(self.label_columns)

    @property
    def data(self):
        return self._data

    @property
    def metric(self):
        return 'r2'

    @property
    def problem_type(self):
        return _REGRESSION


@dataset_registry.register('google_qa_answer_helpful')
class GoogleQuestQALabelHelpful(GoogleQuestQALabel):
    @property
    def label_columns(self):
        return ['answer_helpful']

    @property
    def label_types(self):
        return [_NUMERICAL]


@dataset_registry.register('google_qa_answer_plausible')
class GoogleQuestQALabelPlausible(GoogleQuestQALabel):
    @property
    def label_columns(self):
        return ['answer_plausible']

    @property
    def label_types(self):
        return [_NUMERICAL]


@dataset_registry.register('google_qa_answer_type_procedure')
class GoogleQuestQAAnswerTypeProcedure(GoogleQuestQALabel):
    @property
    def label_columns(self):
        return ['answer_type_procedure']

    @property
    def label_types(self):
        return [_NUMERICAL]


@dataset_registry.register('google_qa_answer_type_reason_explanation')
class GoogleQuestQAAnswerTypeReasonExplanation(GoogleQuestQALabel):
    @property
    def label_columns(self):
        return ['answer_type_reason_explanation']

    @property
    def label_types(self):
        return [_NUMERICAL]


@dataset_registry.register('google_qa_question_type_reason_explanation')
class GoogleQuestQAQuestionTypeReasonExplanation(GoogleQuestQALabel):
    @property
    def label_columns(self):
        return ['question_type_reason_explanation']

    @property
    def label_types(self):
        return [_NUMERICAL]


@dataset_registry.register('google_qa_answer_satisfaction')
class GoogleQuestQAAnswerSatisfaction(GoogleQuestQALabel):
    @property
    def label_columns(self):
        return ['answer_satisfaction']

    @property
    def label_types(self):
        return [_NUMERICAL]


@dataset_registry.register('women_clothing_review')
class WomenClothingReview(BaseMultiModalDataset):
    _SOURCE = 'https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews',
    _INFO = {
        'train': {
            'url': get_repo_url() + 'women_clothing_review/train.pq',
            'sha1sum': '980023e4c063eae51adafc98482610a9a6a1878b'
        },
        'test': {
            'url': get_repo_url() + 'women_clothing_review/test.pq',
            'sha1sum': 'fbc84f757b8a08210a772613ca8342f3990eb1f7'
        }
    }

    def __init__(self, split='train'):
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  'women_clothing_review',
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_parquet(self._path)

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def feature_columns(self):
        return ['Title', 'Review Text', 'Age',
                'Division Name', 'Department Name', 'Class Name']

    @property
    def feature_types(self):
        return [_TEXT, _TEXT, _NUMERICAL, _CATEGORICAL, _CATEGORICAL, _CATEGORICAL]

    @property
    def fill_na_value(self):
        """The default function to fill missing values"""
        return {
            'Division Name': 'None',
            'Department Name': 'None',
            'Class Name': 'None'
        }

    @property
    def label_columns(self):
        return ['Rating']

    @property
    def label_types(self):
        return [_NUMERICAL]

    @property
    def data(self):
        return self._data

    @property
    def metric(self):
        return 'r2'

    @property
    def problem_type(self):
        return _REGRESSION


@dataset_registry.register('melbourne_airbnb')
class MelBourneAirBnb(BaseMultiModalDataset):
    _SOURCE = 'https://www.kaggle.com/tylerx/melbourne-airbnb-open-data',
    _INFO = {
        'train': {
            'url': get_repo_url() + 'airbnb_melbourne/train.pq',
            'sha1sum': '49f7d95df663d1199e6d860102d5863e48765caf'
        },
        'test': {
            'url': get_repo_url() + 'airbnb_melbourne/test.pq',
            'sha1sum': 'c28611514b659295fe4b345c3995005719499946'
        }
    }

    def __init__(self, split='train'):
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  'airbnb_melbourne',
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_parquet(self._path)

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def ignore_columns(self):
        return ['id', 'listing_url', 'scrape_id', 'last_scraped',
                'picture_url', 'host_id', 'host_url', 'host_name',
                'host_thumbnail_url', 'host_picture_url',
                'monthly_price', 'weekly_price', 'price',
                'calendar_last_scraped']

    @property
    def label_columns(self):
        return ['price_label']

    @property
    def feature_columns(self):
        all_columns = sorted(self._data.columns)
        feature_columns = [col for col in all_columns if col not in self.label_columns
                           and col not in self.ignore_columns]
        return feature_columns

    @property
    def data(self):
        return self._data

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def metric(self):
        return "acc"

    @property
    def problem_type(self):
        return _MULTICLASS


@dataset_registry.register('mercari_price_suggestion')
class MercariPriceSuggestion(BaseMultiModalDataset):
    """

    We have converted price to log price by log(1 + price).
    The model is asked to predict the log price.

    """

    _SOURCE = 'https://www.kaggle.com/c/mercari-price-suggestion-challenge',
    _INFO = {
        'train': {
            'url': get_repo_url() + 'mercari_price_suggestion/train.pq',
            'sha1sum': '3c641214bd9010ea2c8f2f5f578436924825bb47'
        },
        'test': {
            'url': get_repo_url() + 'mercari_price_suggestion/dev.pq',
            'sha1sum': 'f0ac64e9cb63f50544bc9379681b512d1dc8a4e0'
        },
        'competition': {
            'url': get_repo_url() + 'mercari_price_suggestion/test.pq',
            'sha1sum': '4f78fee08e04ee8cda3f2a25435e074e981a4a07'
        },
    }

    _LOCAL_NAME = 'mercari_price_suggestion'

    def __init__(self, split='train'):
        super().__init__()
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  self._LOCAL_NAME,
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_parquet(self._path)

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def data(self):
        return self._data

    @property
    def ignore_columns(self):
        return ['train_id', 'price']

    @property
    def label_columns(self):
        return ['log_price']

    @property
    def label_types(self):
        return [_NUMERICAL]

    @property
    def feature_columns(self):
        ret = []
        for col in sorted(self.data.columns):
            if col not in self.ignore_columns and col not in self.label_columns:
                ret.append(col)
        return ret

    @property
    def metric(self):
        return 'r2'

    @property
    def problem_type(self):
        return _REGRESSION

    def postprocess_label(self, data):
        return np.max(np.exp(data) - 1, 0)


@dataset_registry.register('ae_price_prediction')
class AEPricePrediction(BaseMultiModalDataset):
    _SOURCE = 'https://www.kaggle.com/PromptCloudHQ/innerwear-data-from-victorias-secret-and-others'
    _INFO = {
        'train': {
            'url': get_repo_url() + 'ae_price_prediction/train.pq',
            'sha1sum': '5b8a6327cc9429176d58af33ca3cc3480fe6c759'
        },
        'test': {
            'url': get_repo_url() + 'ae_price_prediction/test.pq',
            'sha1sum': '7bebcaae48410386f610fd7a9c37ba0e89602858'
        }
    }

    def __init__(self, split='train'):
        super().__init__()
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  'ae_price_prediction',
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_parquet(self._path)

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def data(self):
        return self._data

    @property
    def ignore_columns(self):
        return ['mrp', 'pdp_url']

    @property
    def feature_columns(self):
        return [col for col in self.data.columns
                if col not in self.ignore_columns and col not in self.label_columns]

    @property
    def label_columns(self):
        return ['price']

    @property
    def label_types(self):
        return [_NUMERICAL]

    @property
    def metric(self):
        return 'r2'

    @property
    def problem_type(self):
        return _REGRESSION


@dataset_registry.register('mercari_price_suggestion100K')
class MercariPriceSuggestion100K(MercariPriceSuggestion):
    _SOURCE = 'https://www.kaggle.com/c/mercari-price-suggestion-challenge',
    _INFO = {
        'train': {
            'url': get_repo_url() + 'mercari_price_suggestion100K/train.pq',
            'sha1sum': '3f28fa2b20ff3ea4cad554012e70abf98db1c620'
        },
        'test': {
            'url': get_repo_url() + 'mercari_price_suggestion100K/test.pq',
            'sha1sum': '25a3cd5a40c695cfae04430475dd970953d06bd7'
        },
    }
    _LOCAL_NAME = 'mercari_price_suggestion100K'

    def __init__(self, split='train'):
        super().__init__(split)


@dataset_registry.register('imdb_genre_prediction')
class IMDBGenrePrediction(BaseMultiModalDataset):
    _SOURCE = 'https://www.kaggle.com/PromptCloudHQ/imdb-data'
    _INFO = {
        'train': {
            'url': get_repo_url() + 'imdb_genre_prediction/train.csv',
            'sha1sum': '56d2d5e3b19663d033fdfb6e33e4eb9c79c67864'
        },
        'test': {
            'url': get_repo_url() + 'imdb_genre_prediction/test.csv',
            'sha1sum': '0e435e917159542d725d21135cfa514ae936d2c1'
        }
    }

    def __init__(self, split='train'):
        super().__init__()
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  'imdb_genre_prediction',
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def data(self):
        return self._data

    @property
    def label_columns(self):
        return ['Genre_is_Drama']

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def feature_columns(self):
        return [col for col in list(self.data.columns) if col not in self.label_columns]

    @property
    def metric(self):
        return 'roc_auc'

    @property
    def problem_type(self):
        return _BINARY


@dataset_registry.register('fake_job_postings')
class FakeJobPostings(BaseMultiModalDataset):
    _SOURCE = 'https://www.kaggle.com/shivamb/real-or-fake-fake-job-posting-prediction'
    _INFO = {
        'train': {
            'url': get_repo_url() + 'fake_job_postings/train.csv',
            'sha1sum': '78c37e46e844c9e268aa8eb6da6168b04a9e6556'
        },
        'test': {
            'url': get_repo_url() + 'fake_job_postings/test.csv',
            'sha1sum': '30fb93d31693083f0a7b22ea006a812f2ae46674'
        }
    }

    def __init__(self, split='train'):
        super().__init__()
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  'fake_job_postings',
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def data(self):
        return self._data

    @property
    def label_columns(self):
        return ['fraudulent']

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def feature_columns(self):
        return [col for col in list(self.data.columns) if col not in self.label_columns]

    @property
    def metric(self):
        return 'roc_auc'

    @property
    def problem_type(self):
        return _BINARY


@dataset_registry.register('kick_starter_funding')
class KickStarterFunding(BaseMultiModalDataset):
    _SOURCE = 'https://www.kaggle.com/codename007/funding-successful-projects?select=train.csv'
    _INFO = {
        'train': {
            'url': get_repo_url() + 'kick_starter_funding/train.csv',
            'sha1sum': 'b0faa47d3e65ff62fbcd105d565b912fce9b004e'
        },
        'test': {
            'url': get_repo_url() + 'kick_starter_funding/test.csv',
            'sha1sum': '454973f5b2866cb27d59f1dff112af9ab2bac8d1'
        }
    }

    def __init__(self, split='train'):
        super().__init__()
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  'kick_starter_funding',
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def data(self):
        return self._data

    @property
    def label_columns(self):
        return ['final_status']

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def feature_columns(self):
        return [col for col in list(self.data.columns) if col not in self.label_columns]

    @property
    def metric(self):
        return 'roc_auc'

    @property
    def problem_type(self):
        return _BINARY


@dataset_registry.register('jc_penney_products')
class JCPennyCategory(BaseMultiModalDataset):
    _SOURCE = 'https://www.kaggle.com/PromptCloudHQ/all-jc-penny-products'
    _INFO = {
        'train': {
            'url': get_repo_url() + 'jc_penney_products/train.csv',
            'sha1sum': 'b59ce843ad05073a3fccf5ebc4840b3b0649f059'
        },
        'test': {
            'url': get_repo_url() + 'jc_penney_products/test.csv',
            'sha1sum': '23bca284354deec13a11ef7bd726d35a01eb1332'
        }
    }

    def __init__(self, split='train'):
        super().__init__()
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  'jc_penney_products',
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def data(self):
        return self._data

    @property
    def label_columns(self):
        return ['sale_price']

    @property
    def label_types(self):
        return [_NUMERICAL]

    @property
    def feature_columns(self):
        return [col for col in list(self.data.columns) if col not in self.label_columns]

    @property
    def metric(self):
        return 'r2'

    @property
    def problem_type(self):
        return _REGRESSION


@dataset_registry.register('wine_reviews')
class WineReviews(BaseMultiModalDataset):
    _SOURCE = 'https://www.kaggle.com/PromptCloudHQ/wine_reviews'
    _INFO = {
        'train': {
            'url': get_repo_url() + 'wine_reviews/train.csv',
            'sha1sum': '8d52f8fd16c36f66812ddebd019a4f96651b9bb0'
        },
        'test': {
            'url': get_repo_url() + 'wine_reviews/test.csv',
            'sha1sum': '9ca88272a403bf3cded29ebd64e07a01b353a903'
        }
    }

    def __init__(self, split='train'):
        super().__init__()
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  'wine_reviews',
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def data(self):
        return self._data

    @property
    def label_columns(self):
        return ['variety']

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def feature_columns(self):
        return [col for col in list(self.data.columns) if col not in self.label_columns]

    @property
    def metric(self):
        return 'acc'

    @property
    def problem_type(self):
        return _MULTICLASS


@dataset_registry.register('news_popularity')
class NewsPopularity(BaseMultiModalDataset):
    _SOURCE = 'https://archive.ics.uci.edu/ml/datasets/online+news+popularity'
    _INFO = {
        'train': {
            'url': get_repo_url() + 'news_popularity/train.csv',
            'sha1sum': '99a8ba552525a84513d1cc16ab056d4703a381cb'
        },
        'test': {
            'url': get_repo_url() + 'news_popularity/test.csv',
            'sha1sum': 'd2928a9a04a3c51a8947b7e86f8f852ab5755c89'
        }
    }

    def __init__(self, split='train'):
        super().__init__()
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  'news_popularity',
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def data(self):
        return self._data

    @property
    def label_columns(self):
        return ['log_shares']

    @property
    def label_types(self):
        return [_NUMERICAL]

    @property
    def feature_columns(self):
        return [col for col in list(self.data.columns) if col not in self.label_columns]

    @property
    def metric(self):
        return 'r2'

    @property
    def problem_type(self):
        return _REGRESSION


@dataset_registry.register('news_channel')
class NewsChannel(BaseMultiModalDataset):
    _SOURCE = 'https://archive.ics.uci.edu/ml/datasets/online+news+popularity'
    _INFO = {
        'train': {
            'url': get_repo_url() + 'news_channel/train.csv',
            'sha1sum': 'ab226210b6a878b449d01f33a195014c65c22311'
        },
        'test': {
            'url': get_repo_url() + 'news_channel/test.csv',
            'sha1sum': 'a71516784ce6e168bd9933e9ec50080f65cb05fd'
        }
    }

    def __init__(self, split='train'):
        super().__init__()
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  'news_channel',
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def data(self):
        return self._data

    @property
    def label_columns(self):
        return ['channel']

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def feature_columns(self):
        return [col for col in list(self.data.columns) if col not in self.label_columns]

    @property
    def metric(self):
        return 'acc'

    @property
    def problem_type(self):
        return _MULTICLASS


@dataset_registry.register('news_popularity2')
class NewsPopularity2(NewsPopularity):
    _SOURCE = 'https://archive.ics.uci.edu/ml/datasets/online+news+popularity'
    _INFO = {
        'train': {
            'url': get_repo_url() + 'news_popularity2/train.csv',
            'sha1sum': '390b15e77fa77a2722ce2d459a977034a9565f46'
        },
        'test': {
            'url': get_repo_url() + 'news_popularity2/test.csv',
            'sha1sum': '297253bdca18f6aafbaee0262be430126c1f9044'
        }
    }

    def __init__(self, split='train'):
        super().__init__()
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  'news_popularity2',
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_csv(self._path)


@dataset_registry.register('fake_job_postings2')
class FakeJobPostings2(FakeJobPostings):
    _SOURCE = 'https://www.kaggle.com/shivamb/real-or-fake-fake-job-posting-prediction'
    _INFO = {
        'train': {
            'url': get_repo_url() + 'fake_job_postings2/train.csv',
            'sha1sum': 'df1c4975fce21e71af552626898e63e6a15a8cfa'
        },
        'test': {
            'url': get_repo_url() + 'fake_job_postings2/test.csv',
            'sha1sum': '021fbed4829eac2c5da002d80c00f1ed339ae420'
        }
    }

    def __init__(self, split='train'):
        super().__init__()
        self._split = split
        self._path = os.path.join(get_data_home_dir(),
                                  'fake_job_postings2',
                                  f'{split}.pq')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_csv(self._path)






