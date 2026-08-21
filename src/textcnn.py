import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: list[int],
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
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
        # x: [B, L] → embed → [B, L, E] → transpose → [B, E, L]
        x = self.embedding(x).transpose(1, 2)

        features = [F.adaptive_max_pool1d(F.relu(conv(x)), 1).squeeze(2) for conv in self.convs]
        x = torch.cat(features, dim=1)
        x = self.dropout(x)
        return self.fc(x)
