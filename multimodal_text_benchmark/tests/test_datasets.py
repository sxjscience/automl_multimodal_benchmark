import pytest
from auto_mm_bench.datasets import *


@pytest.mark.parametrize('key', dataset_registry.list_keys())
def test_generic(key):
    train_dataset = dataset_registry.create(key, 'train')
    test_dataset = dataset_registry.create(key, 'test')
    assert len(train_dataset.label_columns) == len(train_dataset.label_types)
    assert len(test_dataset.label_columns) == len(test_dataset.label_columns)


@pytest.mark.parametrize('split,num_sample',
                         [('train', 1443899),
                          ('test', 360975),
                          ('competition', 97320)])
def test_jigsaw_unintended_bias(split, num_sample):
    df = JigsawUnintendedBias(split).data
    assert len(df) == num_sample


@pytest.mark.parametrize('split,num_sample',
                         [('train', 5091),
                          ('test', 1273),
                          ('competition', 2728)])
def test_product_sentiment(split, num_sample):
    df = MachineHackSentimentPrediction(split).data
    assert len(df) == num_sample


@pytest.mark.parametrize('split,num_sample',
                         [('train', 4863),
                          ('test', 1216),
                          ('competition', 476)])
def test_google_quest_qa(split, num_sample):
    df = GoogleQuestQALabel(split).data
    assert len(df) == num_sample


@pytest.mark.parametrize('split,num_sample',
                         [('train', 18788),
                          ('test', 4698)])
def test_women_clothing_review(split, num_sample):
    df = WomenClothingReview(split).data
    assert len(df) == num_sample


@pytest.mark.parametrize('split,num_sample',
                         [('train', 18316),
                          ('test', 4579)])
def test_melbourne_airbnb(split, num_sample):
    df = MelBourneAirBnb(split).data
    assert len(df) == num_sample


@pytest.mark.parametrize('split,num_sample',
                         [('train', 1186028),
                          ('test', 296507),
                          ('competition', 693359)])
def test_mercari_price_suggestion(split, num_sample):
    df = MercariPriceSuggestion(split).data
    assert len(df) == num_sample


@pytest.mark.parametrize('split,num_sample',
                         [('train', 22662),
                          ('test', 5666)])
def test_ae_price_prediction(split, num_sample):
    df = AEPricePrediction(split).data
    assert len(df) == num_sample
