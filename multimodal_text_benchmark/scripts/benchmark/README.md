# Run experiments

This folder contains the command for run different AutoML methods over the benchmark. To run a method on only a particular dataset (say `salary` for example), you can add the following flag to the below commands: 
``` 
--nickname salary
```

For more information on other flags you can specify, see [ag_benchmark.py](ag_benchmark.py). 

To learn how to add your own ML methods to run on the benchmark, see [h2o_benchmark.py](h2o_benchmark.py) for an example to run H2O AutoML.

## Install Dependencies

### AutoGluon

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
python3 -m pip install autogluon==0.3.0
```

### H2O

We use H2O with version `3.32.0.3`. Thus, to run the H2O experiments, you may use the following command:

```bash
python3 -m pip install h2o==3.32.0.3 py-cpuinfo
```


## Comparing Different Text-only Networks
- roberta_base
    ```bash
    python3 ag_benchmark.py --model ag_text_only --nickname salary --text_backbone roberta_base --decay-rate 1.0 --n-average-epoch 1
    ```
- electra_base
    ```bash
    python3 ag_benchmark.py --model ag_text_only  --nickname salary --text_backbone electra_base --decay-rate 1.0 --n-average-epoch 1
    ```
- electra_base + lr_decay
    ```bash
    python3 ag_benchmark.py --model ag_text_only  --nickname salary --text_backbone electra_base --decay-rate 0.8 --n-average-epoch 1
    ```
- electra_base + lr_decay + average 3
    ```bash
    python3 ag_benchmark.py --model ag_text_only  --nickname salary --text_backbone electra_base --decay-rate 0.8 --n-average-epoch 3 --extract_embedding
    ```

## Comparing Different Fusion Strategies for the Multimodal Text Network
- 
- electra_base + all_text
    ```bash
    python3 ag_benchmark.py --model ag_text_multimodal  --nickname salary --text_backbone electra_base --decay-rate 0.8 --n-average-epoch 3 --multimodal_fusion_strategy all_text --extract_embedding
    ```
- electra_base + fuse-late
    ```bash
    python3 ag_benchmark.py --model ag_text_multimodal  --nickname salary --text_backbone electra_base --decay-rate 0.8 --n-average-epoch 3 --multimodal_fusion_strategy fuse_late --extract_embedding
    ```
- electra_base + fuse-early
    ```bash
    python3 ag_benchmark.py --model ag_text_multimodal  --nickname salary --text_backbone electra_base --decay-rate 0.8 --n-average-epoch 3 --multimodal_fusion_strategy fuse_early --extract_embedding
    ```

## Compare Ensemble Strategies
- weighted
  ```bash
  python3 ag_benchmark.py --model ag_tabular_multimodal  --nickname salary --text_backbone electra_base --decay-rate 0.8 --n-average-epoch 3 --multimodal_fusion_strategy fuse_late --ensemble_option weighted
  ```
- stack (5fold)
  ```bash
  python3 ag_benchmark.py --model ag_tabular_multimodal  --nickname salary --text_backbone electra_base --decay-rate 0.8 --n-average-epoch 3 --multimodal_fusion_strategy fuse_late --ensemble_option 5fold_1stack
  ```
- pre-embedding (pretrained embedding without finetuning)
  ```bash
  python3 ag_benchmark.py --model pre_embedding --nickname salary --ensemble_option weighted
  ```
- text-embedding
  ```bash
  python3 ag_benchmark.py --model tune_embedding --embedding_dir data_scientist_salary_ag_text_only_electra_base_0.8_3_weighted_fuse_late --nickname salary --ensemble_option weighted
  ```
- multimodal-embedding
  ```bash
  python3 ag_benchmark.py --model tune_embedding --embedding_dir data_scientist_salary_ag_text_multimodal_electra_base_0.8_3_weighted_fuse_late  --nickname salary --ensemble_option weighted
  ```

## Other AutoML Solutions
- AutoGluon Stack w/o Multimodal NN
  ```bash
  python3 ag_benchmark.py --model ag_tabular_without_multimodal_nn  --nickname salary --ensemble_option 5fold_1stack
  ```
- AutoGluon Stack w/o Text
  ```bash
  python3 ag_benchmark.py --model ag_tabular_without_text  --nickname salary --ensemble_option 5fold_1stack
  ```
- H2O AutoML
  ```bash
  python3 h2o_benchmark.py --nickname salary --baseline h2o_automl 
  ```
- H2O AutoML + Word2Vec
  ```bash
  python3 h2o_benchmark.py --nickname salary --baseline h2o_word2vec
  ```
- H2O AutoML + Pre-embedding
  ```bash
  python3 h2o_benchmark.py --nickname salary --baseline h2o_embedding --embed_dir salary_pre_embedding
  ```
