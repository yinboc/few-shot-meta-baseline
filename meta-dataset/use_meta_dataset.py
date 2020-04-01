from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import Counter
import gin
import numpy as np
import tensorflow as tf
from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline


BASE_PATH = './materials/records'
GIN_FILE_PATH = './meta_dataset/learn/gin/setups/data_config.gin'

gin.parse_config_file(GIN_FILE_PATH)


def make_md(lst, method, split='train', image_size=126, **kwargs):
    if split == 'train':
        SPLIT = learning_spec.Split.TRAIN
    elif split == 'val':
        SPLIT = learning_spec.Split.VALID
    elif split == 'test':
        SPLIT = learning_spec.Split.TEST

    ALL_DATASETS = lst
    all_dataset_specs = []
    for dataset_name in ALL_DATASETS:
        dataset_records_path = os.path.join(BASE_PATH, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
        all_dataset_specs.append(dataset_spec)

    if method == 'episodic':
        use_bilevel_ontology_list = [False]*len(ALL_DATASETS)
        use_dag_ontology_list = [False]*len(ALL_DATASETS)
        # Enable ontology aware sampling for Omniglot and ImageNet. 
        for i, s in enumerate(ALL_DATASETS):
            if s == 'ilsvrc_2012':
                use_dag_ontology_list[i] = True
            if s == 'omniglot':
                use_bilevel_ontology_list[i] = True
        variable_ways_shots = config.EpisodeDescriptionConfig(
            num_query=None, num_support=None, num_ways=None)

        dataset_episodic = pipeline.make_multisource_episode_pipeline(
            dataset_spec_list=all_dataset_specs,
            use_dag_ontology_list=use_dag_ontology_list,
            use_bilevel_ontology_list=use_bilevel_ontology_list,
            episode_descr_config=variable_ways_shots,
            split=SPLIT, image_size=image_size)
	
        return dataset_episodic

    elif method == 'batch':
        BATCH_SIZE = kwargs['batch_size']
        ADD_DATASET_OFFSET = False
        dataset_batch = pipeline.make_multisource_batch_pipeline(
            dataset_spec_list=all_dataset_specs, batch_size=BATCH_SIZE, split=SPLIT,
            image_size=image_size, add_dataset_offset=ADD_DATASET_OFFSET)
        return dataset_batch

