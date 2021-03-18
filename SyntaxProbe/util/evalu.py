import os
from collections import namedtuple, defaultdict, Counter

import torch as th
from torch import nn
from transformers import BertTokenizer, BertModel

import numpy as np
from scipy.stats import spearmanr

from tqdm import tqdm


def prediction(data_loader, probe, task: str):
    prediction_batches = []
    probe.eval()
    with th.no_grad():
        for batch in tqdm(data_loader, desc="[Pred Batch]"):
            observation_batch, label_batch, length_batch, _ = batch
            observation_batch = observation_batch.to(probe.default_dtype)
            observation_batch = observation_batch.to(probe.device)
            predictions = probe(observation_batch, task=task)
            prediction_batches.append(predictions.cpu())

    return prediction_batches


def predictionLinear(data_loader, probe, task: str, pretrained_bert_dir: str):
    position_embeddings = (
        BertModel.from_pretrained(pretrained_bert_dir)
        .embeddings.position_embeddings.weight.detach()
        .clone()
    )

    prediction_batches = []
    probe.eval()
    with th.no_grad():
        for batch in tqdm(data_loader, desc="[Pred Batch]"):
            observation_batch, label_batch, length_batch, _ = batch

            batch_size = observation_batch.shape[0]
            max_len = observation_batch.shape[1]
            position_ids = th.arange(max_len, dtype=th.long)
            observation_batch = position_embeddings[position_ids]
            observation_batch = observation_batch.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            observation_batch = observation_batch.to(probe.default_dtype)
            observation_batch = observation_batch.to(probe.device)

            predictions = probe(observation_batch, task=task)
            prediction_batches.append(predictions.cpu())

    return prediction_batches


def reportUUAS(prediction_batches, dataset):
    """
    Computes the UUAS score for a dataset.
    From the true and predicted distances, computes a minimum spanning tree
    of each, and computes the percentage overlap between edges in all
    predicted and gold trees.
    All tokens with punctuation part-of-speech are excluded from the minimum
    spanning trees.

    Args:
        prediction_batches: A sequence of batches of predictions for a data split
        dataset: A sequence of batches of Observations
    """

    uspan_total = 0
    uspan_correct = 0
    total_sents = 0

    gold_edge_lens = []
    pred_edge_lens = []
    gold_edge_recall = {}
    gold_edge_cnt = {}
    for (
        prediction_batch,
        (data_batch, label_batch, length_batch, observation_batch),
    ) in tqdm(zip(prediction_batches, dataset), desc="[UUAS]"):
        for prediction, label, length, (observation, _) in zip(
            prediction_batch, label_batch, length_batch, observation_batch
        ):

            words = observation.sentence
            poses = observation.xpos_sentence
            length = int(length)
            assert length == len(observation.sentence)
            prediction = prediction[:length, :length]
            label = label[:length, :length].cpu()

            gold_edges = prims_matrix_to_edges(label, words, poses)
            gold_edges = [tuple(sorted(x)) for x in gold_edges]
            pred_edges = prims_matrix_to_edges(prediction, words, poses)
            pred_edges = [tuple(sorted(x)) for x in pred_edges]

            uspan_correct += len(
                set(gold_edges).intersection(
                    set(pred_edges)
                )
            )
            uspan_total += len(gold_edges)
            total_sents += 1

            # prediction length distribution
            for e in gold_edges:
                gold_edge_lens.append(e[1] - e[0])
            for e in pred_edges:
                pred_edge_lens.append(e[1] - e[0])

            # recall per edge type 
            gold_edges_set = set(str(e[0]) + '-' + str(e[1]) for e in gold_edges)
            for e in pred_edges:
                e_str = str(e[0]) + '-' + str(e[1])
                if(e_str in gold_edges_set):
                    if(observation.head_indices[e[0]] == str(e[1] + 1)):
                        edge_type = observation.governance_relations[e[0]]
                    else: 
                        assert(observation.head_indices[e[1]] == str(e[0] + 1))
                        edge_type = observation.governance_relations[e[1]]
                    if(edge_type in gold_edge_recall): gold_edge_recall[edge_type] += 1
                    else: gold_edge_recall[edge_type] = 1
            for edge_type in observation.governance_relations:
                if(edge_type in gold_edge_cnt): gold_edge_cnt[edge_type] += 1
                else: gold_edge_cnt[edge_type] = 1


    uuas = uspan_correct / float(uspan_total)

    gold_edge_lens = Counter(gold_edge_lens)
    pred_edge_lens = Counter(pred_edge_lens)
    return uuas, gold_edge_lens, pred_edge_lens, gold_edge_recall, gold_edge_cnt


def reportUUASPerLength(prediction_batches, dataset):
    """
    Computes the UUAS score and per length for a dataset.
    From the true and predicted distances, computes a minimum spanning tree
    of each, and computes the percentage overlap between edges in all
    predicted and gold trees.
    All tokens with punctuation part-of-speech are excluded from the minimum
    spanning trees.

    Args:
        prediction_batches: A sequence of batches of predictions for a data split
        dataset: A sequence of batches of Observations
    """
    uspan_total = 0
    uspan_correct = 0
    total_sents = 0
    lengths_to_uuas = defaultdict(list)
    for (
        prediction_batch,
        (data_batch, label_batch, length_batch, observation_batch),
    ) in tqdm(zip(prediction_batches, dataset), desc="[UUAS]"):
        for prediction, label, length, (observation, _) in zip(
            prediction_batch, label_batch, length_batch, observation_batch
        ):

            words = observation.sentence
            poses = observation.xpos_sentence
            length = int(length)
            assert length == len(observation.sentence)
            prediction = prediction[:length, :length]
            label = label[:length, :length].cpu()

            gold_edges = prims_matrix_to_edges(label, words, poses)
            pred_edges = prims_matrix_to_edges(prediction, words, poses)

            corr = len(
                set([tuple(sorted(x)) for x in gold_edges]).intersection(
                    set([tuple(sorted(x)) for x in pred_edges])
                )
            )
            tot = len(gold_edges)
            uspan_correct += corr
            uspan_total += tot
            total_sents += 1

            if tot != 0:
                lengths_to_uuas[length].extend([corr / float(tot)])

    uuas = uspan_correct / float(uspan_total)
    mean_uuas_per_len = {
        length: np.mean(lengths_to_uuas[length]) for length in lengths_to_uuas
    }

    return uuas, mean_uuas_per_len


def reportDistanceSpearmanr(prediction_batches, dataset):
    """
    Writes the Spearman correlations between predicted and true distances.

    For each word in each sentence, computes the Spearman correlation between
    all true distances between that word and all other words, and all
    predicted distances between that word and all other words.

    Computes the average such metric between all sentences of the same length.
    Then computes the average Spearman across sentence lengths 5 to 50.

    Args:
        prediction_batches: A sequence of batches of predictions for a data split
        dataset: A sequence of batches of Observations
    """

    lengths_to_spearmanrs = defaultdict(list)
    for (
        prediction_batch,
        (data_batch, label_batch, length_batch, observation_batch),
    ) in tqdm(zip(prediction_batches, dataset), desc="[DSpr.]"):
        for prediction, label, length, (observation, _) in zip(
            prediction_batch, label_batch, length_batch, observation_batch
        ):

            words = observation.sentence
            length = int(length)
            prediction = prediction[:length, :length]
            label = label[:length, :length].cpu()
            spearmanrs = [
                spearmanr(pred, gold) for pred, gold in zip(prediction, label)
            ]
            lengths_to_spearmanrs[length].extend([x.correlation for x in spearmanrs])

    mean_spearman_for_each_length = {
        length: np.mean(lengths_to_spearmanrs[length])
        for length in lengths_to_spearmanrs
    }
    mean = np.mean(
        [
            mean_spearman_for_each_length[x]
            for x in range(5, 51)
            if x in mean_spearman_for_each_length
        ]
    )

    return mean, mean_spearman_for_each_length


def reportRootAcc(prediction_batches, dataset):
    """
    Computes the root prediction accuracy.

    For each sentence in the corpus, the root token in the sentence
    should be the least deep. This is a simple evaluation.

    Computes the percentage of sentences for which the root token
    is the least deep according to the predicted depths.

    Args:
        prediction_batches: A sequence of batches of predictions for a data split
        dataset: A sequence of batches of Observations
    """
    total_sents = 0
    correct_root_predictions = 0
    for (
        prediction_batch,
        (data_batch, label_batch, length_batch, observation_batch),
    ) in zip(prediction_batches, dataset):
        for prediction, label, length, (observation, _) in zip(
            prediction_batch, label_batch, length_batch, observation_batch
        ):

            length = int(length)
            label = list(label[:length].cpu())
            prediction = prediction.data[:length]
            words = observation.sentence
            poses = observation.xpos_sentence

            correct_root_predictions += label.index(0) == get_nopunct_argmin(
                prediction, words, poses
            )
            total_sents += 1

    root_acc = correct_root_predictions / float(total_sents)
    return root_acc


def reportRootAccPerLength(prediction_batches, dataset):
    """
    Computes the root prediction for each length accuracy.

    For each sentence in the corpus, the root token in the sentence
    should be the least deep. This is a simple evaluation.

    Computes the percentage of sentences for which the root token
    is the least deep according to the predicted depths.

    Args:
        prediction_batches: A sequence of batches of predictions for a data split
        dataset: A sequence of batches of Observations
    """
    total_sents = 0
    correct_root_predictions = 0
    lengths_to_root_acc = defaultdict(list)
    for (
        prediction_batch,
        (data_batch, label_batch, length_batch, observation_batch),
    ) in zip(prediction_batches, dataset):
        for prediction, label, length, (observation, _) in zip(
            prediction_batch, label_batch, length_batch, observation_batch
        ):

            length = int(length)
            label = list(label[:length].cpu())
            prediction = prediction.data[:length]
            words = observation.sentence
            poses = observation.xpos_sentence

            correct_root_predictions += label.index(0) == get_nopunct_argmin(
                prediction, words, poses
            )
            total_sents += 1
            lengths_to_root_acc[length].extend([correct_root_predictions])

    root_acc = correct_root_predictions / float(total_sents)
    mean_root_acc_per_len = {
        length: np.mean(lengths_to_root_acc[length]) for length in lengths_to_root_acc
    }

    return root_acc, mean_root_acc_per_len


def reportDepthSpearmanr(prediction_batches, dataset):
    """
    Writes the Spearman correlations between predicted and true depths.

    For each sentence, computes the spearman correlation between predicted
    and true depths.

    Computes the average such metric between all sentences of the same length.
    Then computes the average Spearman across sentence lengths 5 to 50.

    Args:
        prediction_batches: A sequence of batches of predictions for a data split
        dataset: A sequence of batches of Observations
    """
    lengths_to_spearmanrs = defaultdict(list)
    for (
        prediction_batch,
        (data_batch, label_batch, length_batch, observation_batch),
    ) in zip(prediction_batches, dataset):
        for prediction, label, length, (observation, _) in zip(
            prediction_batch, label_batch, length_batch, observation_batch
        ):

            words = observation.sentence
            length = int(length)
            prediction = prediction[:length]
            label = label[:length].cpu()
            sent_spearmanr = spearmanr(prediction, label)
            lengths_to_spearmanrs[length].append(sent_spearmanr.correlation)

    mean_spearman_for_each_length = {
        length: np.mean(lengths_to_spearmanrs[length])
        for length in lengths_to_spearmanrs
    }

    mean = np.mean(
        [
            mean_spearman_for_each_length[x]
            for x in range(5, 51)
            if x in mean_spearman_for_each_length
        ]
    )

    return mean, mean_spearman_for_each_length


class UnionFind:
    """
    Naive UnionFind implementation for (slow) Prim's MST algorithm
    Used to compute minimum spanning trees for distance matrices
    -----
    author: @john-hewitt
    https://github.com/john-hewitt/structural-probes
    """

    def __init__(self, n):
        self.parents = list(range(n))

    def union(self, i, j):
        if self.find(i) != self.find(j):
            i_parent = self.find(i)
            self.parents[i_parent] = j

    def find(self, i):
        i_parent = i
        while True:
            if i_parent != self.parents[i_parent]:
                i_parent = self.parents[i_parent]
            else:
                break
        return i_parent


def prims_matrix_to_edges(matrix, words, poses):
    """
    Constructs a minimum spanning tree from the pairwise weights in matrix;
    returns the edges.

    Never lets punctuation-tagged words be part of the tree.
    -----
    author: @john-hewitt
    https://github.com/john-hewitt/structural-probes
    """
    pairs_to_distances = {}
    uf = UnionFind(len(matrix))
    for i_index, line in enumerate(matrix):
        for j_index, dist in enumerate(line):
            if poses[i_index] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
                continue
            if poses[j_index] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
                continue
            pairs_to_distances[(i_index, j_index)] = dist
    edges = []
    for (i_index, j_index), distance in sorted(
        pairs_to_distances.items(), key=lambda x: x[1]
    ):
        if uf.find(i_index) != uf.find(j_index):
            uf.union(i_index, j_index)
            edges.append((i_index, j_index))
    return edges


def get_nopunct_argmin(prediction, words, poses):
    """
    Gets the argmin of predictions, but filters out all punctuation-POS-tagged words
    -----
    author: @john-hewitt
    https://github.com/john-hewitt/structural-probes
    """
    puncts = ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
    original_argmin = np.argmin(prediction)
    for i in range(len(words)):
        argmin = np.argmin(prediction)
        if poses[argmin] not in puncts:
            return argmin
        else:
            prediction[argmin] = 9000
    return original_argmin
