"""Dataset and vocabulary utilities for CNN-based text classification."""

from collections import Counter

import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(text: str) -> list[str]:
    """Tokenize text by lowercasing and splitting on whitespace.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return str(text).lower().split()


def build_vocab(
    texts: "list[str] | pd.Series",
    min_freq: int = 2,
    max_vocab_size: int = 30000,
) -> dict[str, int]:
    """Build vocabulary from training texts only.

    Tokens appearing fewer than min_freq times are mapped to <UNK> at inference.
    Vocabulary is capped at max_vocab_size to bound embedding memory.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[word] = len(vocab)

    return vocab


def encode_text(text: str, vocab: dict[str, int], max_length: int = 100) -> list[int]:
    """Convert text to a fixed-length list of token IDs."""
    tokens = tokenize(text)
    unk_index = vocab[UNK_TOKEN]
    pad_index = vocab[PAD_TOKEN]

    token_ids = [vocab.get(token, unk_index) for token in tokens]
    token_ids = token_ids[:max_length]
    token_ids += [pad_index] * (max_length - len(token_ids))

    return token_ids


class TicketDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for IT ticket text classification.

    Encodes text inputs using a fixed vocabulary and returns padded sequences
    with corresponding labels for CNN training and evaluation.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: dict[str, int],
        max_length: int = 100,
    ) -> None:
        """Initialize the dataset.

        Args:
            texts: List of input text strings.
            labels: List of integer class labels.
            vocab: Vocabulary mapping tokens to indices.
            max_length: Maximum sequence length for padding/truncation.
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple of (encoded_text_tensor, label_tensor).
        """
        x = torch.tensor(
            encode_text(self.texts[index], self.vocab, self.max_length), dtype=torch.long
        )
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y


# pandas is only imported for the type annotation — avoid importing at module level
# to keep this module usable without pandas in tests.
try:
    import pandas as pd  # noqa: F401
except ImportError:
    pass
