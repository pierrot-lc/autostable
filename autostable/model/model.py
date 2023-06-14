"""Main model.
It patches the input, computes locations, embed the images
and computes the local and global features.
"""
import torch
import torch.nn as nn

from .encoder import FramesEncoder
from .projector import Projector


class VICRegEncoder(nn.Module):
    def __init__(
        self,
        image_size: int | tuple[int, int],
        n_channels: int,
        n_tokens: int | tuple[int, int],
        hidden_size: int,
        n_heads: int,
        ff_size: int,
        dropout: float,
        n_layers: int,
        projected_size: int,
        n_projected_layers: int,
    ):
        super().__init__()

        self.encoder = FramesEncoder(
            image_size=image_size,
            n_channels=n_channels,
            n_tokens=n_tokens,
            hidden_size=hidden_size,
            n_heads=n_heads,
            ff_size=ff_size,
            dropout=dropout,
            n_layers=n_layers,
        )

        self.projector = Projector(
            embedding_size=hidden_size,
            projected_size=projected_size,
            n_layers=n_projected_layers,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes the input frames.

        ---
        Args:
            x: Input frames.
                Shape of [batch_size, n_channels, image_size, image_size].

        ---
        Returns:
            x_loc: Encoded local tokens of the frames.
                Shape of [batch_size, n_tokens[0] x n_tokens[1], projected_size].
            x_glob: Encoded global token.
                Shape of [batch_size, projected_size].
            x_coords: Absolute coordinates of the tokens.
                Shape of [batch_size, n_tokens[0] x n_tokens[1], 2].
        """
        x, x_coords = self.encoder(x)
        x_loc = self.projector(x)
        x_glob = self.projector(x.mean(dim=1))
        return x_loc, x_glob, x_coords
