"""Predict homographies based on a continuous stream of images.

The stream is processed using a LSTM, so that it can remember what
it has seen previously and predict its homographies based on that.
"""

from typing import Optional

import einops
import torch
import torch.nn as nn
import torchvision.models as models


class SymLog(nn.Module):
    """Compress high range numbers while being the identity
    around [-1, 1].

    See: https://arxiv.org/abs/2301.04104.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sign_x = torch.sign(x)
        abs_x = torch.abs(x)
        return sign_x * torch.log(abs_x + 1)


class SymExp(nn.Module):
    """Output high range numbers while being the identity
    around [-1, 1].

    See: https://arxiv.org/abs/2301.04104.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sign_x = torch.sign(x)
        abs_x = torch.abs(x)
        return sign_x * (torch.exp(abs_x) - 1)


class HomographyModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        resnet_name: str,
        pretrained: bool,
        translation_only: bool,
        resnet_hidden_size: int,
        lstm_hidden_size: int,
        lstm_n_layers: int,
    ):
        super().__init__()
        self.translation_only = translation_only

        assert resnet_name in (
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
        ), "Unknown resnet_name"
        weights = "DEFAULT" if pretrained else None
        self.resnet = models.__dict__[resnet_name](weights=weights)

        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=True
        )

        self.resnet.fc = nn.LazyLinear(resnet_hidden_size)
        self.lstm_module = nn.LSTM(
            input_size=resnet_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_n_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 8, bias=False),
            SymExp(),
        )

    def head_homography(self, x: torch.Tensor) -> torch.Tensor:
        """Estimates the homography matrix directly from the logits.

        ---
        Args:
            x: Logits extracted from the resnet layers.
                Shape of [batch_size, 8].

        ---
        Returns:
            h: Homography matricies.
                Shape of [batch_size, 3, 3].
        """
        z = torch.zeros((x.shape[0], 1)).to(x.device)
        x = torch.concat((x, z), dim=1)  # Add the last element of the matrix.
        x = einops.rearrange(x, "b (h w) -> b h w", h=3, w=3)

        # Bias the homography matrix towards the identity.
        identity = torch.eye(3, device=x.device)
        identity = einops.repeat(identity, "h w -> b h w", b=x.shape[0])
        h = identity + x
        return h

    def forward(self, x: torch.Tensor, state: Optional[tuple] = None) -> tuple:
        """Compute the normalized homography from the given pairs of images.

        ---
        Args:
            x: Batch of consecutive images.
                Shape of [batch_size, n_consecutive_images, n_channels, height, width].
            state: The current state cells of the LSTM.
                Optional, tuple of hidden and cell state.
                If `None`, then the memory is initialized with zeros.
                Shape of [n_layers, batch_size, hidden/cell size].

        ---
        Returns:
            h: Batch of homographies.
                Shape of [batch_size, n_consecutive_images, 3, 3].
            state: The updated state of the LSTM.
                Shape of [n_layers, batch_size, hidden/cell size].
        """
        batch_size = x.shape[0]
        x = einops.rearrange(x, "b s c h w -> (b s) c h w")
        x = self.resnet(x)

        x = einops.rearrange(x, "(b s) h -> b s h", b=batch_size)
        x, state = self.lstm_module(x, state)
        x = self.head(x)
        x = einops.rearrange(x, "b s h -> (b s) h")

        h = self.head_homography(x)

        h = einops.rearrange(h, "(b s) h w -> b s h w", b=batch_size)

        if self.translation_only:
            h = HomographyModel.keep_only_translations(h)

        return h, state

    @staticmethod
    def keep_only_translations(h: torch.Tensor) -> torch.Tensor:
        """Mask the homographies and keep only the translations composites.
        The translations are exprimed by the [2, 0] and [2, 1] elements
        of the homography.
        The rest should stick to the identity matrix.
        It keeps the gradient flowing through the translations.

        ---
        Args
            h: Homographies.
                Shape of [batch_size, seq_len, 3, 3].
        ---
        Returns
            h: Translations only homographies.
                Shape of [batch_size, seq_len, 3, 3].
        """
        # Identity matrix
        h[:, :, 0, 0] = 1
        h[:, :, 1, 1] = 1
        h[:, :, 2, 2] = 1
        h[:, :, 0, 1] = 0
        h[:, :, 1, 0] = 0
        h[:, :, 2, 0] = 0
        h[:, :, 2, 1] = 0

        return h
