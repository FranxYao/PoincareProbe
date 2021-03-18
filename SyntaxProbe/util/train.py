import torch as th
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

import geoopt as gt

from tqdm import tqdm


def train(
    train_data_loader,
    probe,
    loss,
    optimizer,
    task: str,
    dev_data_loader=None,
    scheduler=None,
):
    train_loss, dev_loss = 0, 0

    probe.train()
    for batch in tqdm(train_data_loader, desc="[Train Batch]"):
        optimizer.zero_grad()

        observation_batch, label_batch, length_batch, _ = batch
        observation_batch, label_batch, length_batch = (
            observation_batch.to(probe.default_dtype),
            label_batch.to(probe.default_dtype),
            length_batch.to(probe.default_dtype),
        )
        observation_batch, label_batch, length_batch = (
            observation_batch.to(probe.device),
            label_batch.to(probe.device),
            length_batch.to(probe.device),
        )
        predictions = probe(observation_batch, task=task)
        batch_loss, count = loss(predictions, label_batch, length_batch)
        batch_loss.backward()
        train_loss += batch_loss.item()
        optimizer.step()

    if dev_data_loader is not None:
        probe.eval()
        for batch in tqdm(dev_data_loader, desc="[Dev Batch]"):

            observation_batch, label_batch, length_batch, _ = batch
            observation_batch, label_batch, length_batch = (
                observation_batch.to(probe.default_dtype),
                label_batch.to(probe.default_dtype),
                length_batch.to(probe.default_dtype),
            )
            observation_batch, label_batch, length_batch = (
                observation_batch.to(probe.device),
                label_batch.to(probe.device),
                length_batch.to(probe.device),
            )
            with th.no_grad():
                predictions = probe(observation_batch, task=task)
                batch_loss, count = loss(predictions, label_batch, length_batch)
                dev_loss += batch_loss.item()

        if scheduler is not None:
            scheduler.step(dev_loss)

    return (
        train_loss / len(train_data_loader.dataset),
        dev_loss / len(dev_data_loader.dataset),
    )


def trainLinear(
    train_data_loader,
    probe,
    loss,
    optimizer,
    task: str,
    pretrained_bert_dir: str,
    dev_data_loader=None,
    scheduler=None,
):
    position_embeddings = (
        BertModel.from_pretrained(pretrained_bert_dir)
        .embeddings.position_embeddings.weight.detach()
        .clone()
    )
    train_loss, dev_loss = 0, 0

    probe.train()
    for batch in tqdm(train_data_loader, desc="[Train Batch]"):
        optimizer.zero_grad()

        observation_batch, label_batch, length_batch, _ = batch

        batch_size = observation_batch.shape[0]
        max_len = observation_batch.shape[1]
        position_ids = th.arange(max_len, dtype=th.long)
        observation_batch = position_embeddings[position_ids]
        observation_batch = observation_batch.unsqueeze(0).expand(batch_size, -1, -1)

        observation_batch, label_batch, length_batch = (
            observation_batch.to(probe.default_dtype),
            label_batch.to(probe.default_dtype),
            length_batch.to(probe.default_dtype),
        )
        observation_batch, label_batch, length_batch = (
            observation_batch.to(probe.device),
            label_batch.to(probe.device),
            length_batch.to(probe.device),
        )

        predictions = probe(observation_batch, task=task)
        batch_loss, count = loss(predictions, label_batch, length_batch)
        train_loss += batch_loss.item()

        batch_loss.backward()
        optimizer.step()

    if dev_data_loader is not None:
        probe.eval()
        for batch in tqdm(dev_data_loader, desc="[Dev Batch]"):
            observation_batch, label_batch, length_batch, _ = batch

            batch_size = observation_batch.shape[0]
            max_len = observation_batch.shape[1]
            position_ids = th.arange(max_len, dtype=th.long)
            observation_batch = position_embeddings[position_ids]
            observation_batch = observation_batch.unsqueeze(0).expand(
                batch_size, -1, -1
            )

            observation_batch, label_batch, length_batch = (
                observation_batch.to(probe.default_dtype),
                label_batch.to(probe.default_dtype),
                length_batch.to(probe.default_dtype),
            )
            observation_batch, label_batch, length_batch = (
                observation_batch.to(probe.device),
                label_batch.to(probe.device),
                length_batch.to(probe.device),
            )
            with th.no_grad():
                predictions = probe(observation_batch, task=task)
                batch_loss, count = loss(predictions, label_batch, length_batch)
                dev_loss += batch_loss.item()

        if scheduler is not None:
            scheduler.step(dev_loss)

    return (
        train_loss / len(train_data_loader.dataset),
        dev_loss / len(dev_data_loader.dataset),
    )
