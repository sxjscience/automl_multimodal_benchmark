# Multimodal AutoML Benchmark on Tables with Text Fields

This benchmark contains diverse tabular datasets, each containing numeric/categorical as well as text columns.
The goal is to evaluate the performance of (automated) ML systems for supervised learning tasks (classification and regression) with such multimodal data.
Python code is provided to run different variants of the [AutoGluon](https://github.com/awslabs/autogluon/) AutoML tool on the benchmark.


## Details about the Datasets

The datasets in our benchmark are described in the [multimodal_text_benchmark](multimodal_text_benchmark) folder, as well as example code to load a dataset into Python.


## License
The versions of datasets in this benchmark are released under this license: [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Note that the datasets in this benchmark are [modified versions](multimodal_text_benchmark/scripts/data_processing/README.md) of previously publicly-available original copies and we do not own any of the datasets in the benchmark. Any data from this benchmark which has previously been published elsewhere falls under the original license from which the data originated. 
Please refer to the licenses of each original source linked in the [multimodal_text_benchmark README](multimodal_text_benchmark/README.md).


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
```


## Install AutoGluon

This repository contains a particular version of AutoGluon we previously reported benchmark performance for. We recommend installing it in a fresh virtualenv. 
To use this version, you will need to install MXNet first as a prerequisite. It is recommended to use MXNet 1.8 wheel with CUDA 11.0:

```bash

# CPU-only
python3 -m pip install https://aws-mx-pypi.s3-us-west-2.amazonaws.com/1.8.0/aws_mx-1.8.0-py2.py3-none-manylinux2014_x86_64.whl

# CUDA 11 Version
python3 -m pip install https://aws-mx-pypi.s3-us-west-2.amazonaws.com/1.8.0/aws_mx_cu110-1.8.0-py2.py3-none-manylinux2014_x86_64.whl
```

Once you have MXNet, you can install our version of AutoGluon:

```bash
cd autogluon
bash full_install.sh
```

For more information or if you want to run a different version of AutoGluon, please refer to the [AutoGluon website](https://auto.gluon.ai/). 
Also see the [tutorial on how to easily run AutoGluon on tabular datasets that contain text](https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-multimodal-text-others.html).

## Run Experiments

Go to [multimodal_text_benchmark/scripts/benchmark](multimodal_text_benchmark/scripts/benchmark) to see how to run different ML methods over the benchmark. 

## References

BibTeX entry:

```
@article{agmultimodaltext,
  title={Multimodal AutoML on Structured Tables with Text Fields},
  author={Shi, Xingjian and Mueller, Jonas and Erickson, Nick and Li, Mu and Smola, Alexander},
  journal={8th ICML Workshop on Automated Machine Learning (AutoML)},
  year={2021}
}
```
