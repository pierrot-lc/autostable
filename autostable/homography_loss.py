"""Loss to tell how much two frames are aligned together.
The alignment is done using homographies. Some regularization
can be applied to the homographies to bias the model towards
identity homographies.

Note that it is not mandatory to have standard RGB images.
In fact, we train a model to align latent representations.
"""

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class HomographyLoss(nn.Module):
    def __init__(self, identity_penalty_weight: float = 0.0):
        super().__init__()
        self.identity_penalty_weight = identity_penalty_weight
        self.loss_fn = nn.L1Loss()

    def identity_penalty_loss(self, homographies: torch.Tensor, metrics: dict):
        """Compute the identity penalty loss.
        This bias the model towards identity solutions, and penalyzes
        the model if it outputs homographies that are far away from the
        identity homography.

        ---
        Args:
            homographies: Predicted homographies for this batch of samples.
                Shape of [batch_size, seq_len, 3, 3].
            metrics: Store the loss inside this dictionary.
                Updates the loss.
        """
        identity = torch.eye(3, device=homographies.device)
        identity = einops.repeat(identity, "h w -> b h w", b=homographies.shape[0])
        metrics["identity_penalty_loss"] = F.mse_loss(
            homographies, identity, reduction="mean"
        )
        metrics["loss"] = (
            metrics["loss"]
            + self.identity_penalty_weight * metrics["identity_penalty_loss"]
        )

    def forward(
        self,
        src_to_dest: torch.Tensor,
        dest: torch.Tensor,
        homographies: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the loss.

        ---
        Args:
            src_to_dest: Source image transformed to `dest` image.
                Shape of [batch_size, n_channels, height, width].
            dest: Destination image, the reference to which the source image
                has to be mapped.
                Shape of [batch_size, n_channels, height, width].
            homographies: Predicted homographies for this batch of samples.
                Shape of [batch_size, seq_len, 3, 3].

        ---
        Returns:
            A dictionary with multiple losses.
        """
        metrics = dict()

        main_loss = self.loss_fn(src_to_dest, dest)
        metrics["L1Loss"] = main_loss
        metrics["loss"] = main_loss

        if self.identity_penalty_weight != 0:
            self.identity_penalty_loss(homographies, metrics)

        return metrics

    @staticmethod
    def randomly_select_any_dest(src_to_dest: torch.Tensor) -> torch.Tensor:
        """Choose a random image in the sequence for each element
        of the `src_to_dest` tensor.
        It means that each element of the sequence can be replaced by any element.

        ---
        Args:
            src_to_dest: Contains all transformed images of the sequence.
                Shape of [batch_size, seq_len, n_channels, height, width].

        ---
        Returns:
            selected_dest: New target references.
                Shape of [batch_size, seq_len, n_channels, height, width].
        """
        batch_size, seq_len, n_channels, height, width = src_to_dest.shape

        # Sample random `dest` ids.
        dest_ids = torch.randint(
            low=0, high=seq_len, size=(batch_size, seq_len), device=src_to_dest.device
        )

        # Now select destination images based on the sampled ids.
        dest_ids = einops.repeat(
            dest_ids, "b s -> b s c h w", c=n_channels, h=height, w=width
        )
        src_to_dest = torch.gather(src_to_dest, dim=1, index=dest_ids)

        return src_to_dest
