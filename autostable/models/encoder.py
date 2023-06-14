"""Main encoding model.
"""
import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .projector import Projector


class FramesEncoder(nn.Module):
    """Soft encoder of the input frames.
    The encoding is of the same shape of the input frames.

    It outputs a set of tokens representing patches of the encoded frames,
    used for the VICRegL loss.
    """

    def __init__(
        self,
        image_size: int | tuple[int, int],
        n_channels: int,
        n_tokens: int | tuple[int, int],
        hidden_size: int,
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

        self.project_input = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=hidden_size,
                kernel_size=5,
                padding="same",
            ),
            nn.GELU(),
            nn.BatchNorm2d(hidden_size),
        )
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        hidden_size,
                        hidden_size,
                        kernel_size=5,
                        padding="same",
                    ),
                    nn.GELU(),
                    nn.BatchNorm2d(hidden_size),
                )
                for _ in range(n_layers)
            ]
        )
        self.tok_coordinates = FramesEncoder.token_coordinates(n_tokens)
        self.patchify = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size),
            Rearrange("b c h w -> b (h w) c"),
        )

    def forward(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes the input frames.
        Provides a tokenized version of the frames along with their respective
        coordinates in the frames. Used for the VICRegL loss.

        ---
        Args:
            images: Input images.
                Shape of [batch_size, n_channels, image_size, image_size].

        ---
        Returns:
            frames: Encoded frames.
                Shape of [batch_size, hidden_size, image_size, image_size].
            tokens: Token embeddings.
                Shape of [batch_size, n_tokens[0] x n_tokens[1], hidden_size].
            coords: The absolute position of each token.
                Shape of [batch_size, n_tokens[0] x n_tokens[1], 2].
        """
        # Encode the frames.
        frames = self.project_input(frames)
        for res_layer in self.encoder:
            frames = res_layer(frames) + frames

        # Create the input tokens.
        tokens = self.patchify(frames)

        # Token coordinates.
        coords = self.tok_coordinates.to(frames.device)
        coords = einops.repeat(coords, "n c -> b n c", b=tokens.shape[0])

        return frames, tokens, coords

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


class VICRegEncoder(nn.Module):
    def __init__(
        self,
        image_size: int | tuple[int, int],
        n_channels: int,
        n_tokens: int | tuple[int, int],
        hidden_size: int,
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
        _, x, x_coords = self.encoder(x)
        x_loc = self.projector(x)
        x_glob = self.projector(x.mean(dim=1))
        return x_loc, x_glob, x_coords
