import hashlib

import torch
from loguru import logger


class AttentionModel(torch.nn.Module):
    @logger.catch
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()

        self.embedder = torch.nn.Embedding(
            num_embeddings=8 * 8,
            embedding_dim=8
        )

        self.positional_encoding = torch.linspace(
            start=-1,
            end=1,
            steps=8 * 8
        ).view(1, 8 * 8, 1).repeat(1, 1, 8)

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=8,
                nhead=8,
                dim_feedforward=16,
                dropout=0.2,
                activation="gelu",
                batch_first=True
            ),
            num_layers=4,
            norm=None,
        )

        self.linear = torch.nn.Linear(
            in_features=8 * 8 * 8 + 4 + 1,
            out_features=1
        )

    @logger.catch
    def forward(self, x):
        board, color, castling = x

        # board is 12 * 8 * 8
        # multiply each channel by its index
        indices = torch.arange(12).view(1, 12, 1, 1).to(board.device)
        board = board * indices
        board = torch.sum(board, dim=1)
        board = self.flatten(board)

        emb = self.embedder(board)
        emb = emb + self.positional_encoding

        att = self.encoder(emb)
        flat_att = self.flatten(att)

        x = torch.cat((flat_att, color, castling), dim=1)
        score = self.linear(x)
        return score

    @logger.catch
    def model_hash(self) -> str:
        """Get the hash of the model."""
        return hashlib.md5(
            (str(self.encoder) + str(self.linear)).encode()
        ).hexdigest()
