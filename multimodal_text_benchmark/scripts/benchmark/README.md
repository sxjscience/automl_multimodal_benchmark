# Run experiments

This folder lists various commands to run different ML methods over the benchmark. To run a method on only a particular dataset (say `prod` for example) rather than the full benchmark, you can add the following flag to the below commands: 
``` 
 --dataset product_sentiment_machine_hack
```

For more information on other flags you can specify, see [ag_benchmark.py](ag_benchmark.py). 

To learn how to add your own ML methods to run on the benchmark, see [h2o_benchmark.py](h2o_benchmark.py) for an example to run H2O AutoML.


## AG Tabular-Baselines

- AG-Tabular Weighted, no N-Gram
```
bash run_multimodal_benchmark.sh ag_tabular_without_text no no 123
```

- AG-Tabular Stack, no N-Gram
```
bash run_multimodal_benchmark.sh ag_tabular_without_text 5fold_1stack no 123 0
```

- AG-Tabular Weighted, with N-Gram
```
bash run_multimodal_benchmark.sh ag_tabular_old no no 123 0
```

- AG-Tabular Stack, with N-Gram
```
bash run_multimodal_benchmark.sh ag_tabular_old 5fold_1stack no 123 0
```

## Text-Net

- RoBERTa
```
bash run_multimodal_benchmark.sh ag_text_only no roberta_base_all_text_no_lr_decay_epoch10 123 0
```

- ELECTRA
```
bash run_multimodal_benchmark.sh ag_text_only no electra_base_all_text_e10_no_decay 123 0
```

- ELECTRA + Exponential Decay
```
bash run_multimodal_benchmark.sh ag_text_only no electra_base_all_text_e10 123 0
```


- ELECTRA + Exponential Decay + Averaging
```
bash run_multimodal_benchmark.sh ag_text_only no electra_base_all_text_e10_avg3 123 0
```

## Multimodal-Net

- All-Text
```
bash run_multimodal_benchmark.sh ag_text_multimodal no electra_base_all_text_e10_avg3 123 0
```

- Fuse-Early
```
bash run_multimodal_benchmark.sh ag_text_multimodal no electra_base_early_fusion_e10_avg3 123 0
```

- Fuse-Late, Concat
```
bash run_multimodal_benchmark.sh ag_text_multimodal no electra_base_late_fusion_concat_e10_avg3 123 0
```

- Fuse-Late, Max
```
bash run_multimodal_benchmark.sh ag_text_multimodal no electra_base_late_fusion_max_e10_avg3 123 0
```

- Fuse-Late, Mean
```
bash run_multimodal_benchmark.sh ag_text_multimodal no electra_base_late_fusion_mean_e10_avg3 123 0
```

## Aggregation of Text and Tabular Models

We demonstrate the ensemble-based approach to aggregation. For embedding based approaches, you will need to train the model in the previous step and extract embedding via [extract_text_embeddings.py](extract_text_embeddings.py), and finally run tabular models (e.g. AG Tabular-Baselines, no N-Gram).

- Weighted Ensemble
```
bash run_multimodal_benchmark.sh tabular_multimodal_just_table no electra_base_late_fusion_concat_e10_avg3 123 0
```

- Stacking
```
bash run_multimodal_benchmark.sh tabular_multimodal_just_table 5fold_1stack electra_base_late_fusion_concat_e10_avg3 123 0
```
