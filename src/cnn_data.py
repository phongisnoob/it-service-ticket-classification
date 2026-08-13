import typing
from collections import Counter
from typing import Any

import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(text: Any) -> Any:

    return str(text).lower().split()


def build_vocab(
    texts: Any,
    min_freq: Any = 2,
    max_vocab_size: Any = 30000,
) -> Any:
    """
    Build training set vocab
    2+ = actual words
    """

    counter: typing.Counter[str] = Counter()

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
    text: Any,
    vocab: Any,
    max_length: Any = 100,
) -> Any:
    """
    Text -> fixed-length list of token IDs.
    """

    tokens = tokenize(text)

    unk_index = vocab[UNK_TOKEN]
    pad_index = vocab[PAD_TOKEN]

    token_ids = [vocab.get(token, unk_index) for token in tokens]

    token_ids = token_ids[:max_length]

    padding_needed = max_length - len(token_ids)

    token_ids += [pad_index] * padding_needed

    return token_ids


class TicketDataset(Dataset[typing.Any]):
    def __init__(
        self,
        texts: Any,
        labels: Any,
        vocab: Any,
        max_length: Any = 100,
    ) -> None:

        self.texts = list(texts)
        self.labels = list(labels)

        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> Any:

        return len(self.texts)

    def __getitem__(self, index: Any) -> Any:

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
