# Data Processing Scripts

This directory provides the scripts used to convert the data from each original source into the formatted dataset used in the benchmark. 
These scripts are provided for reproducibility; they are not needed if you just want to run the benchmark or work with some of its datasets. 

For datasets retrieved from Kaggle, you must first install the [Kaggle API](https://www.kaggle.com/docs/api). 


- prod 
  - Name: MachineHack Hackathon: Product Sentiment Analysis, [source](https://www.machinehack.com/hackathons/product_sentiment_classification_weekend_hackathon_19/overview)
  - Commands:
    ```
    mkdir -p machine_hack_product_sentiment
    wget https://automl-mm-bench.s3.amazonaws.com/machine_hack_product_sentiment/all_train.csv -O machine_hack_product_sentiment/all_train.csv
    ```
- airbnb
  - Name: Airbnb Price Prediction, [source](https://www.kaggle.com/tylerx/melbourne-airbnb-open-data)
  - Commands: 
    ```
    wget -O cleansed_listings_dec18.csv https://autogluon-text-data.s3.amazonaws.com/multimodal_text/melbourne-airbnb/cleansed_listings_dec18.csv
    python3 process_airbnb.py
    ```
- salary
  - Name: MachineHack Hackathon: Predict the Data Scientist Salary in India, [source](https://machinehack.com/hackathons/predict_the_data_scientists_salary_in_india_hackathon/overview)
  - Commands:
    ```
    wget PATH_TO_ZIPFILE -O data_scientist_salary/Data.zip
    cd data_scientist_salary
    unzip Data.zip
    cd ..
    python3 process_machine_hack_data_scientist_salary.py --dir_path data_scientist_salary
    ```
- wine
  - Name: Wine Review Classification, [source](https://www.kaggle.com/zynicide/wine-reviews)
  - Commands:
    ```
    python3 process_wine_reviews.py 
    ```
- imdb
  - Name: IMDB Drama or Not Prediction, [source](https://www.kaggle.com/PromptCloudHQ/imdb-data)
  - Commands:
    ```
    python3 process_imdb_genre.py
    ```
- fake
  - Name: Read or Fake Job Posting, [source](https://www.kaggle.com/shivamb/real-or-fake-fake-job-posting-prediction)
  - Commands:
    ```
    python3 process_fake_jobs.py
    ```
- kick
  - Name: Funding Successful Projects on Kickstarter, [source](https://www.kaggle.com/codename007/funding-successful-projects)
  - Commands:
    ```
    python3 process_kickstarter.py
    ```
- jigsaw
  - Name: Jigsaw Unintended Bias in Toxicity Classification
  - Commands:
    ```
    kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
    unzip jigsaw-unintended-bias-in-toxicity-classification.zip -d jigsaw-unintended-bias-in-toxicity-classification
    python3 process_jigsaw.py
    ```
- qaa + qaq
  - Name: Google Quest QA
  - Commands:
    ```
    kaggle competitions download -c google-quest-challenge
    unzip google-quest-challenge.zip -d google-quest-challenge
    python3 process_google_quest_qa.py
    ```
- book
  - Name: MachineHack Hackathon: Book Price Prediction, [source](https://machinehack.com/hackathons/predict_the_price_of_books/overview)
  - Commands:
    ```
    python3 -m pip install openpyxl
    wget PATH_TO_ZIP_FILE -O predict_the_price_of_books/Data.zip
    cd predict_the_price_of_books
    unzip Data.zip
    cd ..
    python3 process_machine_hack_book_price.py --dir_path predict_the_price_of_books
    ```
- jc
  - Name: JCPenney products, [source](https://www.kaggle.com/PromptCloudHQ/all-jc-penny-products)
  - Commands:
    ```
    python3 process_jc_penney.py
    ```
- cloth
  - Name: Women Clothing Review, [source](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews)
  - Commands: 
    ```
    python3 process_women_clothing_review.py
    ```
- ae
  - Name: AE Innerwear Price Prediction, [source](https://www.kaggle.com/PromptCloudHQ/innerwear-data-from-victorias-secret-and-others) 
  - Commands:
    ```
    python3 process_ae_price_prediction.py
    ```
- pop
  - Name: Online popularity of news article, [source](https://archive.ics.uci.edu/ml/datasets/online+news+popularity)
  - Commands:
    ```
    python3 process_news_popularity.py
    ```
- channel
  - Name: Online popularity of news article, [source](https://archive.ics.uci.edu/ml/datasets/online+news+popularity)
  - Commands:
    ```
    python3 process_news_channel.py  # must be executed after process_news_popularity.py 
    ```
- house
  - Name: California House Price Prediction, [source](https://www.kaggle.com/c/california-house-prices)
  - Commands:
    ```
    kaggle competitions download -c california-house-prices
    unzip california-house-prices.zip -d california-house-prices
    python3 process_kaggle_california_house_price.py
    ```
- mercari
  - Name: Mercari Price Suggestion, [source](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data)
  - Commands:
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
