import os
import numpy as np
from auto_mm_bench.datasets import JigsawUnintendedBias, MercariPriceSuggestion


train_dataset = JigsawUnintendedBias('train')
test_dataset = JigsawUnintendedBias('test')
seed = 123

sampled_train_data = train_dataset.data.sample(n=100000, random_state=np.random.RandomState(seed))
sampled_test_data = test_dataset.data.sample(n=int(len(test_dataset.data) / len(train_dataset.data) * 100000),
                                             random_state=seed)
os.makedirs('jigsaw_unintended_bias_sampled')
sampled_train_data.to_parquet(os.path.join('jigsaw_unintended_bias_sampled', 'train.pq'))
sampled_test_data.to_parquet(os.path.join('jigsaw_unintended_bias_sampled', 'test.pq'))


train_dataset = MercariPriceSuggestion('train')
test_dataset = MercariPriceSuggestion('test')
seed = 123

sampled_train_data = train_dataset.data.sample(n=100000, random_state=np.random.RandomState(seed))
sampled_test_data = test_dataset.data.sample(n=int(len(test_dataset.data) / len(train_dataset.data) * 100000),
                                             random_state=seed)
os.makedirs('mercari_price_suggestion_sampled')
sampled_train_data.to_parquet(os.path.join('mercari_price_suggestion_sampled', 'train.pq'))
sampled_test_data.to_parquet(os.path.join('mercari_price_suggestion_sampled', 'test.pq'))
