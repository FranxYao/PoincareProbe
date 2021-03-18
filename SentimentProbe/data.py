import os

import torch as th
import torch.nn as nn
import torch.utils as tutils
from torch.utils.data import TensorDataset, DataLoader
# import torchtext as thtext
from transformers import BertTokenizer, BertModel

import numpy as np


def preprocess(text, tokenizer):
    text_ipt = tokenizer(
        text, padding=True, truncation=True, max_length=64, return_tensors="pt",
    )

    return text_ipt


if __name__ == "__main__":
    bert_pretrained_file = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_pretrained_file, do_lower_case=True)

    data_path = "./data/rt-polaritydata/rt-polaritydata/"
    neg_file = os.path.join(data_path, "rt-polarity.neg")
    pos_file = os.path.join(data_path, "rt-polarity.pos")

    neg_text, pos_text = [], []

    with open(neg_file, "r", encoding="latin-1") as f:
        for line in f:
            neg_text.append(line.strip())

    with open(pos_file, "r", encoding="latin-1") as f:
        for line in f:
            pos_text.append(line.strip())

    neg_len = [len(text.split()) for text in neg_text]
    pos_len = [len(text.split()) for text in pos_text]

    print(f"Average length: {np.mean(neg_len + pos_len):.2f}")
    print(f"Max length: {np.max(neg_len + pos_len)}")

    neg_text_ipt = preprocess(neg_text, tokenizer)
    neg_label = th.LongTensor([0] * len(neg_text))
    pos_text_ipt = preprocess(pos_text, tokenizer)
    pos_label = th.LongTensor([1] * len(pos_text))

    neg_dataset = TensorDataset(
        neg_text_ipt["input_ids"],
        neg_text_ipt["token_type_ids"],
        neg_text_ipt["attention_mask"],
        neg_label,
    )
    pos_dataset = TensorDataset(
        pos_text_ipt["input_ids"],
        pos_text_ipt["token_type_ids"],
        pos_text_ipt["attention_mask"],
        pos_label,
    )

    dataset = neg_dataset + pos_dataset
    dataset_len = len(dataset)
    train_dataset_len = 8528
    dev_dataset_len = 1067
    test_dataset_len = 1067

    train_dataset, dev_dataset, test_dataset = tutils.data.random_split(
        dataset, [train_dataset_len, dev_dataset_len, test_dataset_len]
    )

    print(f"Saving train/dev/test data to ./data")
    th.save(train_dataset, os.path.join("data", "train_dataset.pt"))
    th.save(train_dataset, os.path.join("data", "dev_dataset.pt"))
    th.save(test_dataset, os.path.join("data", "test_dataset.pt"))
