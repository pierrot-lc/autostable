"""Project embeddings for the contrastive loss."""
import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, embedding_size: int, projected_size: int, n_layers: int):
        """Initialize the projector.

        ---
        Args:
            embedding_size: Size of the input embeddings.
            projected_size: Size of the projected embeddings.
            n_layers: Number of layers.
        """
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(embedding_size, projected_size),
            nn.ReLU(),
            nn.LayerNorm(projected_size),
        )
        self.mlp = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(projected_size, projected_size),
                    nn.ReLU(),
                    nn.LayerNorm(projected_size),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the embeddings.

        ---
        Args:
            x: Input embeddings.
                Shape of [batch_size, embedding_size].

        ---
        Returns:
            x: Projected embeddings.
                Shape of [batch_size, projected_size].
        """
        x = self.projector(x)
        x = self.mlp(x)
        return x
