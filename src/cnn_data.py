from collections import Counter

import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(text):

    return str(text).lower().split()


def build_vocab(
    texts,
    min_freq=2,
    max_vocab_size=30000,
):
    """
    Build training set vocab
    2+ = actual words
    """

    counter = Counter()

    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for word, frequency in counter.most_common():

        if frequency < min_freq:
            continue

        if len(vocab) >= max_vocab_size:
            break

        vocab[word] = len(vocab)

    return vocab


def encode_text(
    text,
    vocab,
    max_length=100,
):
    """
    Text -> fixed-length list of token IDs.
    """

    tokens = tokenize(text)

    unk_index = vocab[UNK_TOKEN]
    pad_index = vocab[PAD_TOKEN]

    token_ids = [
        vocab.get(token, unk_index)
        for token in tokens
    ]

    token_ids = token_ids[:max_length]

    padding_needed = max_length - len(token_ids)

    token_ids += [pad_index] * padding_needed

    return token_ids


class TicketDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        vocab,
        max_length=100,
    ):

        self.texts = list(texts)
        self.labels = list(labels)

        self.vocab = vocab
        self.max_length = max_length


    def __len__(self):

        return len(self.texts)


    def __getitem__(self, index):

        encoded_text = encode_text(
            self.texts[index],
            self.vocab,
            self.max_length,
        )

        x = torch.tensor(
            encoded_text,
            dtype=torch.long,
        )

        y = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )

        return x, y