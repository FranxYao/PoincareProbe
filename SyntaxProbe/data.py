import sys
import os
from collections import namedtuple, defaultdict

import torch as th
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

import numpy as np

import dill
import h5py
from tqdm import tqdm

from util.data import *
from util.task import ParseDistanceTask, ParseDepthTask


if __name__ == "__main__":
    data_path = "./data"
    train_data_path = os.path.join(data_path, "ptb3-wsj-train.conllx")
    dev_data_path = os.path.join(data_path, "ptb3-wsj-dev.conllx")
    test_data_path = os.path.join(data_path, "ptb3-wsj-test.conllx")

    train_hdf5_path = "./data/embeddings/raw.train.bertbase-layers.hdf5"
    dev_hdf5_path = "./data/embeddings/raw.dev.bertbase-layers.hdf5"
    test_hdf5_path = "./data/embeddings/raw.test.bertbase-layers.hdf5"

    model_name = "bert-base-cased"
    layer_index = 7
    task_name = "distance"

    train_dataset_path = os.path.join(
        "./data/dataset/",
        task_name,
        model_name,
        "train-layer-" + str(layer_index) + ".pt",
    )
    dev_dataset_path = os.path.join(
        "./data/dataset/",
        task_name,
        model_name,
        "dev-layer-" + str(layer_index) + ".pt",
    )
    test_dataset_path = os.path.join(
        "./data/dataset/",
        task_name,
        model_name,
        "test-layer-" + str(layer_index) + ".pt",
    )

    train_text = loadText(train_data_path)
    dev_text = loadText(dev_data_path)
    test_text = loadText(test_data_path)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name)
    bert.cuda()
    bert.eval()

    if model_name == "bert-base-cased":
        LAYER_COUNT = 13
        FEATURE_COUNT = 768
    else:
        LAYER_COUNT = 25
        FEATURE_COUNT = 1024

    # NOTE: only call these functions once 
    # saveBertHDF5(
    #     train_hdf5_path, train_text, tokenizer, bert, LAYER_COUNT, FEATURE_COUNT
    # )
    # saveBertHDF5(dev_hdf5_path, dev_text, tokenizer, bert, LAYER_COUNT, FEATURE_COUNT)
    # saveBertHDF5(test_hdf5_path, test_text, tokenizer, bert, LAYER_COUNT, FEATURE_COUNT)

    observation_fieldnames = [
        "index",
        "sentence",
        "lemma_sentence",
        "upos_sentence",
        "xpos_sentence",
        "morph",
        "head_indices",
        "governance_relations",
        "secondary_relations",
        "extra_info",
        "embeddings",
    ]

    observation_class = get_observation_class(observation_fieldnames)

    train_observations = load_conll_dataset(train_data_path, observation_class)
    dev_observations = load_conll_dataset(dev_data_path, observation_class)
    test_observations = load_conll_dataset(test_data_path, observation_class)

    train_observations = embedBertObservation(
        train_hdf5_path, train_observations, tokenizer, observation_class, layer_index
    )
    dev_observations = embedBertObservation(
        dev_hdf5_path, dev_observations, tokenizer, observation_class, layer_index
    )
    test_observations = embedBertObservation(
        test_hdf5_path, test_observations, tokenizer, observation_class, layer_index
    )

    if task_name == "distance":
        task = ParseDistanceTask()
    else:
        task = ParseDepthTask()

    train_dataset = ObservationIterator(train_observations, task)
    dev_dataset = ObservationIterator(dev_observations, task)
    test_dataset = ObservationIterator(test_observations, task)

    th.save(train_dataset, train_dataset_path, pickle_module=dill)
    th.save(dev_dataset, dev_dataset_path, pickle_module=dill)
    th.save(test_dataset, test_dataset_path, pickle_module=dill)
