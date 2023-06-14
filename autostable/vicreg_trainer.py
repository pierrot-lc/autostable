from collections import defaultdict
from typing import Any

import einops
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from vicreg_loss import VICRegLLoss

from .models import VICRegEncoder


class VICRegTrainer:
    def __init__(
        self,
        model: VICRegEncoder,
        loss_fn: VICRegLLoss,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        device: str,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.device = device

    def compute_metrics(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """

        ---
        Args:
            batch: A batch of pairs of frames.
                Shape of [batch_size, 2, n_channels, height, width].
        """
        batch_size = batch.shape[0]
        batch = einops.rearrange(batch, "b f c h w -> (b f) c h w")
        loc, glob, coords = self.model(batch)

        # Compute the loss.
        loc = einops.rearrange(loc, "(b f) t d -> b f t d", b=batch_size)
        glob = einops.rearrange(glob, "(b f) d -> b f d", b=batch_size)
        coords = einops.rearrange(coords, "(b f) t d -> b f t d", b=batch_size)
        metrics = self.loss_fn(
            loc[:, 0],
            loc[:, 1],
            glob[:, 0],
            glob[:, 1],
            coords[:, 0].float(),
            coords[:, 1].float(),
        )
        return metrics

    def train_model(self):
        self.model.train()
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            batch = batch.to(self.device)
            metrics = self.compute_metrics(batch)
            self.optimizer.zero_grad()
            metrics["loss"].backward()
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, torch.Tensor]:
        self.model.eval()
        loader_metrics = defaultdict(list)
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(self.device)
            metrics = self.compute_metrics(batch)
            for key, value in metrics.items():
                loader_metrics[key].append(value.cpu().item())

        loader_metrics = {
            key: torch.tensor(value).mean() for key, value in loader_metrics.items()
        }
        return loader_metrics

    def launch_training(self, config: dict[str, Any], eval_every: int = 10):
        self.model.to(self.device)

        for epoch_id in tqdm(range(self.n_epochs), desc="Epoch"):
            self.train_model()

            if epoch_id % eval_every == 0:
                for loader_type, loader in [
                    ("train", self.train_loader),
                    ("validation", self.val_loader),
                ]:
                    metrics = self.evaluate(loader)
                torch.save(self.model.state_dict(), "encoder.pth")
