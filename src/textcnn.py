import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_filters,
        kernel_sizes,
        num_classes,
        dropout=0.3,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Linear(
            num_filters * len(kernel_sizes),
            num_classes,
        )


    def forward(self, x):

        # [B, L]
        x = self.embedding(x)

        # [B, L, E]
        # ->
        # [B, E, L]
        x = x.transpose(1, 2)

        features = []

        for conv in self.convs:

            feature = F.relu(
                conv(x)
            )

            feature = F.adaptive_max_pool1d(
                feature,
                output_size=1,
            )

            feature = feature.squeeze(2)

            features.append(feature)

        x = torch.cat(
            features,
            dim=1,
        )

        x = self.dropout(x)

        return self.fc(x)
