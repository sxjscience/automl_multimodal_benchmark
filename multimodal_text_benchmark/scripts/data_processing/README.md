# Data Processing Scripts

This directory provides the scripts used to convert the data from each original source into the formatted dataset used in the benchmark. 
These scripts are provided for reproducibility; they are not needed if you just want to run the benchmark or work with some of its datasets. 

For datasets retrieved from Kaggle, you must first install the [Kaggle API](https://www.kaggle.com/docs/api). 

Statistics of the datasets may be generated as follows: 

```
python3 get_dataset_analytics.py
```


## Jigsaw Unintended Bias in Toxicity Classification

```
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
unzip jigsaw-unintended-bias-in-toxicity-classification.zip -d jigsaw-unintended-bias-in-toxicity-classification
python3 process_jigsaw.py
```

## Product Sentiment Analysis

```
mkdir -p machine_hack_product_sentiment
wget https://automl-mm-bench.s3.amazonaws.com/machine_hack_product_sentiment/all_train.csv -O machine_hack_product_sentiment/all_train.csv
```

## Google Quest QA

```
kaggle competitions download -c google-quest-challenge
unzip google-quest-challenge.zip -d google-quest-challenge
python3 process_google_quest_qa.py
```

## Women Clothing Review

```
python3 process_women_clothing_review.py
```

## Airbnb Price Prediction

```
wget -O cleansed_listings_dec18.csv https://autogluon-text-data.s3.amazonaws.com/multimodal_text/melbourne-airbnb/cleansed_listings_dec18.csv
python3 process_airbnb.py
```


## Mercari Price Suggestion Challenge

```
kaggle competitions download -c mercari-price-suggestion-challenge
unzip mercari-price-suggestion-challenge.zip -d mercari-price-suggestion-challenge
sudo apt-get install p7zip-full
cd mercari-price-suggestion-challenge
7z x train.tsv.7z
7z x test.tsv.7z
cd ..
python3 process_mercari_price_suggestion.py
```

## AE Innerwear Price Prediction

```
wget -O ae_com.csv.zip https://automl-mm-bench.s3.amazonaws.com/ae_com.csv.zip
unzip ae_com.csv.zip
python3 process_ae_price_prediction.py
```

## JC Penny

Source: https://www.kaggle.com/PromptCloudHQ/all-jc-penny-products

First download the data and then run:

```
python3 process_jc_penney.py
```

## Kickstarter Funding

Source: https://www.kaggle.com/codename007/funding-successful-projects?select=train.csv

First download the data and then run:

```
python3 process_kicksarter.py
```

## IMDB

Source: https://www.kaggle.com/PromptCloudHQ/imdb-data?select=IMDB-Movie-Data.csv

First download the data and then run:

```
python3 process_imdb_genre.py
```

## News Popularity

Source: https://archive.ics.uci.edu/ml/datasets/online+news+popularity

First download the data and then run:

```
python3 process_news_popularity.py
```

## Wine Review

Retrieve data from: https://www.kaggle.com/zynicide/wine-reviews (winemag-data-130k-v2.csv)

Then run: 

```
python3 process_wine_reviews.py 
```

## Fake Jobs

Source: https://www.kaggle.com/shivamb/real-or-fake-fake-job-posting-prediction

First download the data and then run:

```
python3 process_fake_jobs.py
```
