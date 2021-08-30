import pytest
from auto_mm_bench.datasets import *
from auto_mm_bench.datasets import create_dataset

@pytest.mark.parametrize('key', dataset_registry.list_keys())
def test_generic(key):
    train_dataset = dataset_registry.create(key, 'train')
    test_dataset = dataset_registry.create(key, 'test')
    assert len(train_dataset.label_columns) == len(train_dataset.label_types)
    assert len(test_dataset.label_columns) == len(test_dataset.label_columns)
    for col in train_dataset.label_columns:
        assert col not in train_dataset.feature_columns


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


@pytest.mark.parametrize('split,num_sample',
                         [('train', 15841),
                          ('test', 3961),
                          ('competition', 6601)])
def test_data_scientist_salary(split, num_sample):
    df = create_dataset('data_scientist_salary', split).data
    assert len(df) == num_sample


@pytest.mark.parametrize('split,num_sample',
                         [('train', 4989),
                          ('test', 1248),
                          ('competition', 1560)])
def test_bookprice_prediction(split, num_sample):
    df = create_dataset('bookprice_prediction', split).data
    assert len(df) == num_sample


@pytest.mark.parametrize('split,num_sample',
                         [('train', 37951),
                          ('test', 9488),
                          ('competition', 31626)])
def test_california_house_price(split, num_sample):
    df = create_dataset('california_house_price', split).data
    assert len(df) == num_sample
