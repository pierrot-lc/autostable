"""Main encoding model.
"""
import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer

from .projector import Projector


class FramesEncoder(nn.Module):
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
    ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if isinstance(n_tokens, int):
            n_tokens = (n_tokens, n_tokens)

        assert (
            image_size[0] % n_tokens[0] == 0
        ), "The image height is not a multiple of n_tokens."
        assert (
            image_size[1] % n_tokens[1] == 0
        ), "The image width is not a multiple of n_tokens."
        kernel_size = (
            image_size[0] // n_tokens[0],
            image_size[1] // n_tokens[1],
        )

        self.tok_coordinates = FramesEncoder.token_coordinates(n_tokens)
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                stride=kernel_size,
            ),
            nn.GELU(),
            nn.LayerNorm([hidden_size, n_tokens[0], n_tokens[1]]),
            Rearrange("b c h w -> b h w c"),
        )
        self.positional_encoding = nn.Sequential(
            Summer(PositionalEncoding2D(hidden_size)),
            Rearrange("b h w c -> b (h w) c"),
        )
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_size, n_heads, ff_size, dropout, batch_first=True
            ),
            n_layers,
        )

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input frames.

        ---
        Args:
            images: Input images.
                Shape of [batch_size, n_channels, image_size, image_size].

        ---
        Returns:
            x: Token embeddings.
                Shape of [batch_size, n_tokens[0] x n_tokens[1], hidden_size].
            coords: The absolute position of each token.
                Shape of [batch_size, n_tokens[0] x n_tokens[1], 2].
        """
        # Create the input tokens.
        # To shape of [batch_size, n_tokens, n_tokens, hidden_size].
        tokens = self.patch_embedding(frames)
        # To shape of [batch_size, n_tokens x n_tokens, hidden_size].
        tokens = self.positional_encoding(tokens)
        # Main ViT backbone.
        tokens = self.backbone(tokens)

        # Token coordinates.
        coords = self.tok_coordinates.to(frames.device)
        coords = einops.repeat(coords, "n c -> b n c", b=tokens.shape[0])

        return tokens, coords

    @staticmethod
    def token_coordinates(n_tokens: tuple[int, int]) -> torch.Tensor:
        """Returns the position of each token.

        ---
        Args:
            n_tokens: Number of tokens.
                Tuple of [n_tokens_height, n_tokens_width].

        ---
        Returns:
            The absolute coordinates of each token.
                Shape of [n_tokens[0] x n_tokens[1], 2].
        """
        row_indices = torch.arange(n_tokens[0])
        col_indices = torch.arange(n_tokens[1])
        x, y = torch.meshgrid(col_indices, row_indices, indexing="xy")
        coordinates = torch.stack((y, x), dim=-1)
        coordinates = einops.rearrange(coordinates, "h w c -> (h w) c")
        return coordinates

    @staticmethod
    def tokens_to_image(tokens: torch.Tensor) -> torch.Tensor:
        """Converts the tokens back to an image."""
        pass


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
