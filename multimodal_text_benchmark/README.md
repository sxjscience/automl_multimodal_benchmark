# Datasets in the Benchmark

Here is how to load a particular dataset into a pandas DataFrame. The following code loads the `prod` (i.e. product_sentiment_machine_hack) dataset:

```python
from auto_mm_bench.datasets import dataset_registry

print(dataset_registry.list_keys())  # use these keys to specify which dataset to load

train_dataset = dataset_registry.create('product_sentiment_machine_hack', 'train')
test_dataset = dataset_registry.create('product_sentiment_machine_hack', 'test')
print(train_dataset.data)
```

These commands will also download a local copy of the dataset file, which is either in Parquet or CSV format (both can be easily loaded into pandas DataFrame).
The data files are currently hosted in AWS Simple Cloud Storage (S3), you can find the location of each file in [datasets.py](src/auto_mm_bench/datasets.py). 
For the official benchmark, each dataset is provided with a training/test split, but we leave it up to ML systems how much validation data to set aside from the training data. 

The statistics of the benchmarking datasets are listed in the table below. The list of `keys` are also shown in [scripts/benchmark/benchmark_datasets.txt](scripts/benchmark/benchmark_datasets.txt). You can use these `key` strings to load any of the datasets in the benchmark via the above code. 


| ID       | key |  #Train | #Test | Task | Metric  | Prediction Target |
|----------|-----|---------|-------|------|---------|-------------------|
| prod     | product_sentiment_machine_hack  | 5,091 | 1,273 | multiclass | accuracy | sentiment related to product |
| airbnb   | melbourne_airbnb  | 18,316  | 4,579  | multiclass  | accuracy | price of Airbnb listing |
| channel  | news_channel  | 20,284  | 5,071  | multiclass | accuracy | category of news article |
| wine     | wine_reviews  | 84,123  | 21,031 | multiclass | accuracy | variety of wine |
| imdb     | imdb_genre_prediction | 800 | 200 | binary | roc_auc | whether film is a drama |
| jigsaw   | jigsaw_unintended_bias100K | 100,000 | 25,000 | binary | roc_auc | whether comments are toxic |
| fake     | fake_job_postings2 | 12,725 | 3,182 | binary | roc_auc | whether job postings are fake |
| kick     | kick_starter_funding | 86,052 | 21,626 | binary | roc_auc | will Kickstarter get funding |
| ae       | ae_price_prediction  | 22,662 | 5,666 | regression | r2 | American-Eagle item prices |
| qaa      | google_qa_answer_type_reason_explanation | 4,863 | 1,216 | regression | r2 | type of answer |
| qaq      | google_qa_question_type_reason_explanation | 4,863 | 1,216 | regression | r2 | type of question |
| cloth    | women_clothing_review | 18,788 | 4,698 | regression | r2 | review score |
| mercari  | mercari_price_suggestion100K | 100,000 | 25,000 | regression | r2 | price of Mercari products |
| jc       | jc_penney_products | 10,860 | 2,715 | regression | r2 | price of JC Penney products |
| pop      | news_popularity2 | 24,007 | 6,002 | regression | r2 | news article popularity online |


## Benchmark Creation

The folder [scripts/data_processing](scripts/data_processing/README.md) contains the scripts previously used to create the benchmark version of each dataset from the original data source. 

## Additional Details for each Dataset

### prod

Original data source: Product Sentiment Classification (MachineHack prediction competition)

Link: https://machinehack.com/hackathons/product_sentiment_classification_weekend_hackathon_19/overview

**Task:** (Multiclass Classification) Classify the sentiment of user reviews of products based on the review text and product type.
 
### airbnb

Original data source: Melbourne Airbnb Open Data

Link: https://www.kaggle.com/tylerx/melbourne-airbnb-open-data

**Task:** (Multiclass Classification) Predict the price label of AirBnb listings (in Melbourne, Australia) based on the page of each listing which includes many miscellaneous features about the listing.

### wine

Original data source: 130k wine reviews with variety, location, winery, price, and description (from WineEnthusiast)

Link: https://www.kaggle.com/zynicide/wine-reviews

**Task:** (Multiclass Classification)  Classify the variety of wines based on tasting descriptions from sommeliers, their price, country-of-origin, and other features.

### imdb

Original data source: IMDB data from 2006 to 2016 - A data set of 1,000 popular movies on IMDB in the last 10 years

Link: https://www.kaggle.com/PromptCloudHQ/imdb-data

**Note:** PromptCloud released the original version of the data from which the version of this dataset in our benchmark
was created.

**Task:** (Binary Classification) Predict whether or not a movie falls within the Drama category based on its name, description,
actors/directors, year released, runtime, and other features.

### jigsaw

Original data source: Jigsaw Unintended Bias in Toxicity Classification (Kaggle prediction competition)

Link: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification

**Task:** (Binary Classification) Predict whether online social media comments are toxic based on their text and additional tabular features providing information about the post (e.g. likes, rating, date created, etc.). 

### fake

Original data source: Fake JobPosting Prediction

Link: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction

**Task:** (Binary Classification) Predict whether online job postings are real or fake based on their text, amount of salary offered, degree of education demanded, etc.

### kick

Original data source: Funding Successful Projects on Kickstarter

Link: https://www.kaggle.com/codename007/funding-successful-projects

**Task:** (Binary Classification) Predict whether a proposed Kickstarter project will achieve funding goal based on its title, description, amount of money requested, date posted, and other features.

### ae

Original data source: Innerwear Data from American Eagle

Link: https://www.kaggle.com/PromptCloudHQ/innerwear-data-from-victorias-secret-and-others

**Note:** PromptCloud released the original version of the data from which the version of this dataset in our benchmark
was created.

**Task:** (Regression) Predict the price of inner-wear items sold by retailer American Eagle based on features from their online product page.

### qaa

Original data source: Google QUEST Q&A Labeling (Kaggle prediction competition)

Link: https://www.kaggle.com/c/google-quest-challenge

**Task:** (Regression) Given a question and an answer (from the Crowdsource team at Google) as well as additional category features, predict the (subjective) type of the answer in relation to the question.

### qaq

Original data source: Google QUEST Q&A Labeling (Kaggle prediction competition)

Link: https://www.kaggle.com/c/google-quest-challenge

**Task:** (Regression) Given a question and an answer (from the Crowdsource team at Google) as well as additional
 category features, predict the (subjective) type of the question in relation to the answer.

### cloth

Original data source: Women's E-Commerce Clothing Reviews

Link: https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

**Task:** Predict the score of a customer review of clothing items (sold by an anonymous retailer) based on the review text, and product features like the clothing category. 

### mercari

Original data source: Mercari Price Suggestion Challenge (Kaggle prediction competition)

Link: https://www.kaggle.com/c/mercari-price-suggestion-challenge

**Task:** (Regression) Predict the price of items sold in the online marketplace of Mercari based on miscellaneous information from the product page like name, description, free shipping, etc.

### jc

Original data source: 20,000 product listings from JCPenney

Link: https://www.kaggle.com/PromptCloudHQ/all-jc-penny-products

**Note:** PromptCloud released the original version of the data from which the version of this dataset in our benchmark
was created.

**Task:** (Regression) Predict the sale price of items sold on the website of the retailer JC Penney based on miscellaneous information on the product page like its title, description, rating, etc.

### pop

Original data source: Online News Popularity Data Set

Link: https://archive.ics.uci.edu/ml/datasets/online+news+popularity

**Task:** (Regression) Predict the popularity (number of shares on social media, on log-scale) of Mashable.com news articles based on the text of their title, as well as auxiliary numerical features like the number of words in the article, its average token length, and how many keywords are listed, etc.

### channel

Original data source: Online News Popularity Data Set

Link: https://archive.ics.uci.edu/ml/datasets/online+news+popularity

**Task:** (Multiclass Classification) Predict which news category (i.e. channel) a Mashable.com news article belongs to based on the text of its title, as well as auxiliary numerical features like the number of words in the article, its average token length, how many keywords are listed, etc.
