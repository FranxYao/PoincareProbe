import sys

sys.path.append("..")

import os
from collections import namedtuple, defaultdict

import torch as th
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np

import h5py
from tqdm import tqdm


def dataStat(data):
    """
    Statistics of text.
    """

    len_min, len_max, len_avg = float("Inf"), 0, 0
    text_len = []

    for text in data:
        text_len.append(len(text.split(" ")))

        if len_min > len(text):
            len_min = len(text)
        if len_max < len(text):
            len_max = len(text)
        len_avg += len(text)

    len_avg /= len(data)
    text_len = np.array(text_len)

    return len_min, len_max, len_avg, text_len


def loadText(data_path):
    """
    Yields batches of lines describing a sentence in conllx.

    Args:
        data_path: Path of a conllx file.
    Yields:
        a list of lines describing a single sentence in conllx.
    """
    text, t = [], []

    for line in tqdm(open(data_path)):
        if line.startswith("#"):
            continue

        if not line.strip():
            text += [" ".join(t)]
            t = []
        else:
            t.append(line.split("\t")[1])

    return text


def saveBertHDF5(path, text, tokenizer, model, LAYER_COUNT, FEATURE_COUNT):
    """
    Takes raw text and saves BERT-cased features for that text to disk
    Adapted from the BERT readme (and using the corresponding package) at
    https://github.com/huggingface/pytorch-pretrained-BERT
    """

    model.eval()
    with h5py.File(path, "w") as fout:
        for index, line in tqdm(enumerate(text)):
            line = line.strip()  # Remove trailing characters
            line = "[CLS] " + line + " [SEP]"
            tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(line)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segment_ids = [1 for x in tokenized_text]

            # Convert inputs to PyTorch tensors
            tokens_tensor = th.tensor([indexed_tokens]).cuda()
            segments_tensors = th.tensor([segment_ids]).cuda()

            with th.no_grad():
                encoded_layers = model(
                    tokens_tensor, segments_tensors, output_hidden_states=True
                )
                # embeddings + 12 layers
                encoded_layers = encoded_layers[-1]
            dset = fout.create_dataset(
                str(index), (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT)
            )
            dset[:, :, :] = np.vstack([x.cpu().numpy() for x in encoded_layers])


def embedBertObservation(
    hdf5_path, observations, tokenizer, observation_class, layer_index
):
    """
    Adds pre-computed BERT embeddings from disk to Observations.
    
    Reads pre-computed subword embeddings from hdf5-formatted file.
    Sentences should be given integer keys corresponding to their order
    in the original file.
    Embeddings should be of the form (layer_count, subword_sent_length, feature_count)
    subword_sent_length is the length of the sequence of subword tokens
    when the subword tokenizer was given each canonical token (as given
    by the conllx file) independently and tokenized each. Thus, there
    is a single alignment between the subword-tokenized sentence
    and the conllx tokens.

    Args:
        hdf5_path: The filepath of a hdf5 file containing embeddings.
        observations: A list of Observation objects composing a dataset.
        tokenizer: (optional) a tokenizer used to map from
            conllx tokens to subword tokens.
        layer_index: The index corresponding to the layer of representation
            to be used. (e.g., 0 for BERT embeddings, 1, ..., 12 for BERT 
            layer 1, ..., 12)

    Returns:
        A list of Observations with pre-computed embedding fields.
        
    Raises:
        AssertionError: sent_length of embedding was not the length of the
        corresponding sentence in the dataset.
    """

    hf = h5py.File(hdf5_path, "r")
    indices = list(hf.keys())

    single_layer_features_list = []

    for index in tqdm(sorted([int(x) for x in indices]), desc="[aligning embeddings]"):
        observation = observations[index]
        feature_stack = hf[str(index)]
        single_layer_features = feature_stack[layer_index]
        tokenized_sent = tokenizer.wordpiece_tokenizer.tokenize(
            "[CLS] " + " ".join(observation.sentence) + " [SEP]"
        )
        untokenized_sent = observation.sentence
        untok_tok_mapping = match_tokenized_to_untokenized(
            tokenized_sent, untokenized_sent
        )
        assert single_layer_features.shape[0] == len(tokenized_sent)
        single_layer_features = th.tensor(
            [
                np.mean(
                    single_layer_features[
                        untok_tok_mapping[i][0] : untok_tok_mapping[i][-1] + 1, :
                    ],
                    axis=0,
                )
                for i in range(len(untokenized_sent))
            ]
        )
        assert single_layer_features.shape[0] == len(observation.sentence)
        single_layer_features_list.append(single_layer_features)

    embeddings = single_layer_features_list
    embedded_observations = []
    for observation, embedding in zip(observations, embeddings):
        embedded_observation = observation_class(*(observation[:-1]), embedding)
        embedded_observations.append(embedded_observation)

    return embedded_observations


def embedELMoObservation(hdf5_path, observations, observation_class, layer_index):
    """
    Adds pre-computed ELMo embeddings from disk to Observations.
    
    Reads pre-computed subword embeddings from hdf5-formatted file.
    Sentences should be given integer keys corresponding to their order
    in the original file.
    Embeddings should be of the form (layer_count, subword_sent_length, feature_count)
    subword_sent_length is the length of the sequence of subword tokens
    when the subword tokenizer was given each canonical token (as given
    by the conllx file) independently and tokenized each. Thus, there
    is a single alignment between the subword-tokenized sentence
    and the conllx tokens.

    Args:
        hdf5_path: The filepath of a hdf5 file containing embeddings.
        observations: A list of Observation objects composing a dataset.
        tokenizer: (optional) a tokenizer used to map from
            conllx tokens to subword tokens.
        layer_index: The index corresponding to the layer of representation
            to be used. (e.g., 0 for BERT embeddings, 1, ..., 12 for BERT 
            layer 1, ..., 12)

    Returns:
        A list of Observations with pre-computed embedding fields.
        
    Raises:
        AssertionError: sent_length of embedding was not the length of the
        corresponding sentence in the dataset.
    """

    hf = h5py.File(hdf5_path, "r")
    indices = filter(lambda x: x != "sentence_to_index", list(hf.keys()))
    single_layer_features_list = []
    for index in tqdm(sorted([int(x) for x in indices]), desc="[aligning embeddings]"):
        observation = observations[index]
        feature_stack = hf[str(index)]
        single_layer_features = feature_stack[layer_index]
        assert single_layer_features.shape[0] == len(observation.sentence)
        single_layer_features_list.append(single_layer_features)

    embeddings = single_layer_features_list
    embedded_observations = []
    for observation, embedding in zip(observations, embeddings):
        embedded_observation = observation_class(*(observation[:-1]), embedding)
        embedded_observations.append(embedded_observation)

    return embedded_observations


def get_observation_class(fieldnames):
    """
    Returns a namedtuple class for a single observation.

    The namedtuple class is constructed to hold all language and annotation
    information for a single sentence or document.

    Args:
        fieldnames: a list of strings corresponding to the information in each
            row of the conllx file being read in. (The file should not have
            explicit column headers though.)
    Returns:
        A namedtuple class; each observation in the dataset will be an instance
        of this class.
    """
    return namedtuple("Observation", fieldnames)


def generate_lines_for_sent(lines):
    """
    Yields batches of lines describing a sentence in conllx.

    Args:
        lines: Each line of a conllx file.
    Yields:
        a list of lines describing a single sentence in conllx.
    """

    buf = []
    for line in lines:
        if line.startswith("#"):
            continue
        if not line.strip():
            if buf:
                yield buf
                buf = []
            else:
                continue
        else:
            buf.append(line.strip())
    if buf:
        yield buf


def load_conll_dataset(filepath, observation_class):
    """
    Reads in a conllx file; generates Observation objects
    
    For each sentence in a conllx file, generates a single Observation
    object.

    Args:
        filepath: the filesystem path to the conll dataset

    Returns:
        A list of Observations 
    """
    observations = []

    lines = (x for x in open(filepath))
    for buf in generate_lines_for_sent(lines):
        conllx_lines = []
        for line in buf:
            conllx_lines.append(line.strip().split("\t"))
        embeddings = [None for x in range(len(conllx_lines))]
        observation = observation_class(*zip(*conllx_lines), embeddings)
        observations.append(observation)

    return observations


def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
    """
    Aligns tokenized and untokenized sentence given subwords "##" prefixed

    Assuming that each subword token that does not start a new word is prefixed
    by two hashes, "##", computes an alignment between the un-subword-tokenized
    and subword-tokenized sentences.

    Args:
        tokenized_sent: a list of strings describing a subword-tokenized sentence
        untokenized_sent: a list of strings describing a sentence, no subword tok.
    Returns:
        A dictionary of type {int: list(int)} mapping each untokenized sentence
        index to a list of subword-tokenized sentence indices
    """
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 1
    while untokenized_sent_index < len(untokenized_sent) and tokenized_sent_index < len(
        tokenized_sent
    ):

        while tokenized_sent_index + 1 < len(tokenized_sent) and tokenized_sent[
            tokenized_sent_index + 1
        ].startswith("##"):

            mapping[untokenized_sent_index].append(tokenized_sent_index)
            tokenized_sent_index += 1

        mapping[untokenized_sent_index].append(tokenized_sent_index)
        untokenized_sent_index += 1
        tokenized_sent_index += 1

    return mapping


def custom_pad(batch_observations, use_disk_embeddings=True):
    """
    Pads sequences with 0 and labels with -1; used as collate_fn of DataLoader.
    
    Loss functions will ignore -1 labels.
    If labels are 1D, pads to the maximum sequence length.
    If labels are 2D, pads all to (maxlen,maxlen).
    Args:
        batch_observations: A list of observations composing a batch
    
    Return:
        A tuple of:
            input batch, padded
            label batch, padded
            lengths-of-inputs batch, padded
            Observation batch (not padded)
    """
    if use_disk_embeddings:
        if hasattr(batch_observations[0][0].embeddings, "device"):
            seqs = [x[0].embeddings.clone().detach() for x in batch_observations]
        else:
            seqs = [th.Tensor(x[0].embeddings) for x in batch_observations]
    else:
        seqs = [x[0].sentence for x in batch_observations]
    lengths = th.tensor([len(x) for x in seqs])
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    label_shape = batch_observations[0][1].shape
    maxlen = int(max(lengths))
    label_maxshape = [maxlen for x in label_shape]
    labels = [-th.ones(*label_maxshape) for x in seqs]
    for index, x in enumerate(batch_observations):
        length = x[1].shape[0]
        if len(label_shape) == 1:
            labels[index][:length] = x[1]
        elif len(label_shape) == 2:
            labels[index][:length, :length] = x[1]
        else:
            raise ValueError(
                "Labels must be either 1D or 2D right now; got either 0D or >3D"
            )
    labels = th.stack(labels)

    return seqs, labels, lengths, batch_observations


class ObservationIterator(Dataset):
    """
    List Container for lists of Observations and labels for them.
    Used as the iterator for a PyTorch dataloader.
    -----
    author: @john-hewitt
    https://github.com/john-hewitt/structural-probes
    """

    def __init__(self, observations, task):
        self.observations = observations
        self.set_labels(observations, task)

    def set_labels(self, observations, task):
        """
        Constructs aand stores label for each observation.

        Args:
            observations: A list of observations describing a dataset
            task: a Task object which takes Observations and constructs labels.
        """
        self.labels = []
        for observation in tqdm(observations, desc="[computing labels]"):
            self.labels.append(task.labels(observation))

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.labels[idx]
