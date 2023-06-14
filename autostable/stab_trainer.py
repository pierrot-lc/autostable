"""Instantiate all composants and train the model.

The training is monitored using `Visdom`.
"""

from collections import defaultdict

import einops
import numpy as np
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

    def compute_metrics(self, x: torch.Tensor) -> tuple:
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
        src = einops.rearrange(x, "b s c h w -> (b s) c h w")

        # Compute and apply homography
        homography, _ = self.homography_model(x)
        homography = einops.rearrange(homography, "b s h w -> (b s) h w")
        src_to_dest = self.homography_warper(src, homography)

        # First image is its own reference.
        # For the rest of the sequence, the reference is always the previous image.
        y = einops.rearrange(src_to_dest, "(b s) c h w -> b s c h w", b=x.shape[0])
        dest = self.loss_fn.randomly_select_any_dest(y)

        dest = dest.detach()  # Do not adapt to the future.
        dest = einops.rearrange(dest, "b s c h w -> (b s) c h w")

        # Compute loss
        metrics.update(self.loss_fn(src_to_dest, dest, homography))

        # Store some images
        images = {
            "src": src.cpu(),
            "dest": dest.cpu(),
            "src_to_dest": torch.clamp(src_to_dest, 0, 1).cpu(),
        }

        return metrics["loss"], metrics, images

    @torch.no_grad()
    def eval_loader(self, loader: DataLoader):
        self.homography_model.to(self.device)
        self.loss_fn.to(self.device)
        self.homography_model.eval()
        metrics = defaultdict(list)

        for x in loader:
            x = x.to(self.device)
            _, m, _ = self.compute_metrics(x)
            for name, value in m.items():
                metrics[name].append(value.cpu().item())

        return {name: np.mean(values) for name, values in metrics.items()}

    @torch.no_grad()
    def plot_eval(self):
        for loader, is_train in zip(
            [self.train_loader, self.test_loader], [True, False]
        ):
            # Lower the number of samples for fast evaluation.
            num_samples = loader.sampler._num_samples
            loader.sampler._num_samples = num_samples // 10

            # Logs metrics.
            metrics = self.eval_loader(loader)
            for name, value in metrics.items():
                self.visdom.add_data(value, name, is_train)

            # Put back the original number of samples.
            loader.sampler._num_samples = num_samples

            # Save model state if better.
            if not is_train and self.best_loss > metrics["loss"]:
                self.best_loss = metrics["loss"]
                self.best_state_dict = self.homography_model.state_dict()

            # Logs predictions.
            x = next(iter(loader))
            x = x.to(self.device)
            _, _, images = self.compute_metrics(x)

            image_id = torch.randint(0, len(images["src"]), (1,)).item()
            image_id = int(image_id)
            images = {k: i[image_id] for k, i in images.items()}  # Select images.

            zeros_channel = torch.zeros_like(images["src"], device=images["src"].device)
            src_red = torch.cat((images["src"], zeros_channel, zeros_channel), dim=0)
            dest_green = torch.cat(
                (zeros_channel, images["dest"], zeros_channel), dim=0
            )
            src_to_dest_white = torch.cat(
                (images["src_to_dest"], images["src_to_dest"], images["src_to_dest"]),
                dim=0,
            )

            images = torch.stack(
                [
                    src_red,
                    dest_green,
                    (src_to_dest_white + src_red) / 2,
                    (dest_green + src_to_dest_white) / 2,
                ],
                dim=0,
            ).cpu()  # Build tensor of images.
            self.visdom.add_images(images, "(src, dest, diff_src, diff_dest)", is_train)

        self.visdom.update()

    def eval(self):
        self.visdom.add_gradient_flow(self.homography_model)
        self.plot_eval()
        self.save_state()

    def train(self):
        self.homography_model.to(self.device)
        self.loss_fn.to(self.device)
        self.best_loss = float("+inf")

        for _ in tqdm(range(self.train_config["n_epochs"]), position=0):
            self.homography_model.train()

            for x in tqdm(self.train_loader, position=1):
                x = x.to(self.device)
                self.optimizer.zero_grad()

                loss, _, _ = self.compute_metrics(x)

                loss.backward()
                clip_grad_norm_(
                    self.homography_model.parameters(), 0.5
                )  # Training an RNN leads to gradient vanishing and exploding problems.
                self.optimizer.step()

            self.eval()
