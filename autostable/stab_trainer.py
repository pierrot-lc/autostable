"""Instantiate all composants and train the model.

The training is monitored using `Visdom`.
"""

from collections import defaultdict
from typing import Any

import einops
import torch
from kornia.geometry import HomographyWarper
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .homography_loss import HomographyLoss
from .models import FramesEncoder, HomographyModel


class Trainer:
    def __init__(
        self,
        homography_model: HomographyModel,
        frames_encoder: FramesEncoder,
        loss_fn: HomographyLoss,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        image_size: list[int],
        n_epochs: int,
        device: str,
    ):
        self.homography_model = homography_model
        self.frames_encoder = frames_encoder
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.device = device

        self.homography_warper = HomographyWarper(image_size[0], image_size[1])

    def compute_metrics(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Do a batch-forward and compute relevant metrics.

        ---
        Args
            x: Batch of consecutive images.
                Shape of [batch_size, sequence_len, n_channels, height, width].

        ---
        Returns:
            loss: Main loss to which the backward should be called.
            metrics: Dictionary with a bunch of metrics to plot.
            images: Dictionary of some useful images to plot.
        """
        metrics = dict()
        batch_size = x.shape[0]

        # Compute and apply homography.
        homography, _ = self.homography_model(x)
        homography = einops.rearrange(homography, "b s h w -> (b s) h w")
        x = einops.rearrange(x, "b s c h w -> (b s) c h w")
        with torch.no_grad():
            src, _, _ = self.frames_encoder(x)  # Take the encoded frames.
        src_to_dest = self.homography_warper(src, homography)

        # Select random destination frames.
        y = einops.rearrange(src_to_dest, "(b s) c h w -> b s c h w", b=batch_size)
        dest = HomographyLoss.randomly_select_any_dest(y)
        dest = dest.detach()  # The gradient only pass through one element of the pairs.
        dest = einops.rearrange(dest, "b s c h w -> (b s) c h w")

        # Compute loss.
        metrics.update(self.loss_fn(src_to_dest, dest, homography))

        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, torch.Tensor]:
        self.homography_model.eval()
        loader_metrics = defaultdict(list)

        for batch in loader:
            batch = batch.to(self.device)
            metrics = self.compute_metrics(batch)
            for name, value in metrics.items():
                loader_metrics[name].append(value.cpu().item())

        loader_metrics = {
            key: torch.tensor(value).mean() for key, value in loader_metrics.items()
        }
        return loader_metrics

    def train_model(self):
        self.homography_model.train()

        for batch in tqdm(self.train_loader, desc="Batch", leave=False):
            batch = batch.to(self.device)
            metrics = self.compute_metrics(batch)
            self.optimizer.zero_grad()
            metrics["loss"].backward()
            # Training an RNN leads to gradient vanishing and exploding problems.
            clip_grad_norm_(self.homography_model.parameters(), 0.5)
            self.optimizer.step()

    def launch_training(self, config: dict[str, Any], eval_every: int = 10):
        self.homography_model.to(self.device)
        self.frames_encoder.to(self.device)
        self.loss_fn.to(self.device)
        self.frames_encoder.eval()

        for epoch_id in tqdm(range(self.n_epochs), desc="Epoch"):
            self.train_model()

            if epoch_id % eval_every == 0:
                for loader_type, loader in [
                    ("train", self.train_loader),
                    ("validation", self.val_loader),
                ]:
                    metrics = self.evaluate(loader)
                torch.save(self.homography_model.state_dict(), "homography.pth")
