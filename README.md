# Benchmarking Multimodal AutoML for Tabular Data with Text Fields 

Repository for the NeurIPS 2021 Dataset Track Submission "Benchmarking Multimodal AutoML for Tabular Data with Text Fields" ([Link](https://openreview.net/forum?id=Q0zOIaec8HF), [Full Paper with Appendix](https://openreview.net/attachment?id=Q0zOIaec8HF&name=supplementary_material)). 
An earlier version of the paper, called "Multimodal AutoML on Structured Tables with Text Fields" ([Link](https://openreview.net/forum?id=OHAIVOOl7Vl)) has been accepted by ICML 2021 AutoML workshop as **Oral**. 
As we have updated the benchmark with more datasets, the version used in the AutoML workshop paper is archived at the [icml_workshop branch](https://github.com/sxjscience/automl_multimodal_benchmark/tree/icml_workshop). 

This benchmark contains a diverse collection of tabular datasets. Each dataset contains numeric/categorical as well as text columns.
The goal is to evaluate the performance of (automated) ML systems for supervised learning tasks (classification and regression) with such multimodal data.
Python code is provided to run different variants of the [AutoGluon](https://github.com/awslabs/autogluon/) AutoML tool and [H2O](https://github.com/h2oai/h2o-3) on the benchmark.

## Details about the Datasets

Here's the brief description of the datasets in our benchmark. The details are described in detail in the [multimodal_text_benchmark](multimodal_text_benchmark) folder.


| ID       | key |  #Train | #Test | Task | Metric  | Prediction Target |
|----------|-----|---------|-------|------|---------|-------------------|
| prod     | product_sentiment_machine_hack  | 5,091 | 1,273 | multiclass | accuracy | sentiment related to product |
| salary   | data_scientist_salary  | 15,84 | 3961 | multiclass | accuracy | salary range in data scientist job listings |
| airbnb   | melbourne_airbnb  | 18,316  | 4,579  | multiclass  | accuracy | price of Airbnb listing |
| channel  | news_channel  | 20,284  | 5,071  | multiclass | accuracy | category of news article |
| wine     | wine_reviews  | 84,123  | 21,031 | multiclass | accuracy | variety of wine |
| imdb     | imdb_genre_prediction | 800 | 200 | binary | roc_auc | whether film is a drama |
| fake     | fake_job_postings2 | 12,725 | 3,182 | binary | roc_auc | whether job postings are fake |
| kick     | kick_starter_funding | 86,052 | 21,626 | binary | roc_auc | will Kickstarter get funding |
| jigsaw   | jigsaw_unintended_bias100K | 100,000 | 25,000 | binary | roc_auc | whether comments are toxic |
| qaa      | google_qa_answer_type_reason_explanation | 4,863 | 1,216 | regression | r2 | type of answer |
| qaq      | google_qa_question_type_reason_explanation | 4,863 | 1,216 | regression | r2 | type of question |
| book     | bookprice_prediction | 4,989 | 1,248 | regression | r2 | price of books |
| jc       | jc_penney_products | 10,860 | 2,715 | regression | r2 | price of JC Penney products |
| cloth    | women_clothing_review | 18,788 | 4,698 | regression | r2 | review score |
| ae       | ae_price_prediction  | 22,662 | 5,666 | regression | r2 | American-Eagle item prices |
| pop      | news_popularity2 | 24,007 | 6,002 | regression | r2 | news article popularity online |
| house    | california_house_price | 24,007 | 6,002 | regression | r2 | sale price of houses in California |
| mercari  | mercari_price_suggestion100K | 100,000 | 25,000 | regression | r2 | price of Mercari products |

## License
The versions of datasets in this benchmark are released under the [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license.
Note that the datasets in this benchmark are [modified versions](multimodal_text_benchmark/scripts/data_processing/README.md) of previously publicly-available original copies and we **do not own** any of the datasets in the benchmark. 
Any data from this benchmark which has previously been published elsewhere falls under the original license from which the data originated. 
Please refer to the licenses of each original source linked in the [multimodal_text_benchmark/README.md](multimodal_text_benchmark/README.md).


## Install the Benchmark Suite

```bash
cd multimodal_text_benchmark
# Install the benchmarking suite
python3 -m pip install -U -e .
```

You can do a quick test of the installation by going to the test folder

```bash
cd multimodal_text_benchmark/tests
python3 -m pytest test_datasets.py
```

To access one dataset, try to use the following code:

```python
from auto_mm_bench.datasets import dataset_registry

print(dataset_registry.list_keys())
train_dataset = dataset_registry.create('product_sentiment_machine_hack', 'train')
test_dataset = dataset_registry.create('product_sentiment_machine_hack', 'test')
print(train_dataset.data)
print(test_dataset.data)
```

To access the 18 datasests used in the benchmark 

```python
from auto_mm_bench.datasets import create_dataset, TEXT_BENCHMARK_ALIAS_MAPPING

for dataset_name in list(TEXT_BENCHMARK_ALIAS_MAPPING.values()):
    print(dataset_name)
    dataset = create_dataset(dataset_name)
```


### Install AutoGluon
 
We use the official AutoGluon for benchmark purpose. Please refer to the [AutoGluon website](https://auto.gluon.ai/) for installation instructions.
Also see the [tutorial on how to easily run AutoGluon on tabular datasets that contain text](https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-multimodal-text-others.html).

Following is the example installation commands that we used in our benchmark. First, install MXNet 1.8 wheel with CUDA 11.0:

```bash

# CPU-only
python3 -m pip install https://aws-mx-pypi.s3-us-west-2.amazonaws.com/1.8.0/aws_mx-1.8.0-py2.py3-none-manylinux2014_x86_64.whl

# CUDA 11 Version
python3 -m pip install https://aws-mx-pypi.s3-us-west-2.amazonaws.com/1.8.0/aws_mx_cu110-1.8.0-py2.py3-none-manylinux2014_x86_64.whl
```

Once you have MXNet, install AutoGluon with the following command:

```bash
python3 -m pip install autogluon
```
## Run Experiments

Go to [multimodal_text_benchmark/scripts/benchmark](multimodal_text_benchmark/scripts/benchmark) to see how to run different ML methods over the benchmark. 

## References

BibTeX entry of the ICML Workshop Version:

```
@article{agmultimodaltext,
  title={Multimodal AutoML on Structured Tables with Text Fields},
  author={Shi, Xingjian and Mueller, Jonas and Erickson, Nick and Li, Mu and Smola, Alexander},
  journal={8th ICML Workshop on Automated Machine Learning (AutoML)},
  year={2021}
}
```
