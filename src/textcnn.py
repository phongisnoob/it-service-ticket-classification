"""TextCNN model architecture for text classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """TextCNN model for sentence classification using multi-size convolutional filters.

    This implementation uses parallel 1D convolutions with different kernel sizes
    to capture n-gram features at multiple scales, followed by max pooling and
    a fully connected classification layer.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: list[int],
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the TextCNN model.

        Args:
            vocab_size: Size of the vocabulary (number of unique tokens).
            embedding_dim: Dimensionality of word embeddings.
            num_filters: Number of output channels for each convolutional layer.
            kernel_sizes: List of kernel sizes for parallel convolutions.
            num_classes: Number of output classes for classification.
            dropout: Dropout probability applied after concatenating features.
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, sequence_length] containing token indices.

        Returns:
            Output logits tensor of shape [batch_size, num_classes].
        """
        # x: [B, L] → embed → [B, L, E] → transpose → [B, E, L]
        x = self.embedding(x).transpose(1, 2)

        features = [F.adaptive_max_pool1d(F.relu(conv(x)), 1).squeeze(2) for conv in self.convs]
        x = torch.cat(features, dim=1)
        x = self.dropout(x)
        return self.fc(x)  # type: ignore[no-any-return]
