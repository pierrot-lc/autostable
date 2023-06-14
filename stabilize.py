from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, RandomSampler, random_split
from torchinfo import summary

from autostable import (
    ContinuousDataset,
    FramesEncoder,
    HomographyLoss,
    HomographyModel,
    StabTrainer,
    VICRegEncoder,
)


def init_dataloaders(config: DictConfig) -> tuple[DataLoader, DataLoader]:
    folder_path = to_absolute_path(config.data.folder_path)
    test_size = config.data.test_size
    batch_size = config.trainer.batch_size
    n_batches = config.trainer.n_batches

    # Create datasets.
    dataset = ContinuousDataset.from_folder(
        Path(folder_path), config.data.image_size, config.data.n_frames
    )
    train_set, val_set = random_split(dataset, lengths=[1 - test_size, test_size])

    # Create data loaders.
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=RandomSampler(
            train_set,
            replacement=True,
            num_samples=n_batches * batch_size,
        ),
        num_workers=config.trainer.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=RandomSampler(
            val_set,
            replacement=True,
            num_samples=n_batches * batch_size,
        ),
        num_workers=config.trainer.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def init_encoder(config: DictConfig) -> FramesEncoder:
    model = VICRegEncoder(
        config.data.image_size,
        config.data.n_channels,
        config.encoder.n_tokens,
        config.encoder.hidden_size,
        config.encoder.n_layers,
        config.encoder.projected_size,
        config.encoder.n_projected_layers,
    )

    encoder = model.encoder
    # Print model summary.
    summary(
        encoder,
        input_size=(
            config.trainer.batch_size,
            config.data.n_channels,
            config.data.image_size[0],
            config.data.image_size[1],
        ),
        device=config.device,
    )

    return encoder


def init_model(config: DictConfig) -> HomographyModel:
    return HomographyModel(
        config.data.n_channels,
        config.homography.resnet.name,
        config.homography.resnet.pretrained,
        config.homography.translation_only,
        config.homography.hidden_size,
        config.homography.hidden_size,
        config.homography.n_lstm_layers,
    )


def init_loss(config: DictConfig) -> HomographyLoss:
    return HomographyLoss(config.homography.identity_penalty)


def init_trainer(
    model: HomographyModel,
    encoder: FramesEncoder,
    loss_fn: HomographyLoss,
    optimizer: Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: DictConfig,
) -> StabTrainer:
    return StabTrainer(
        model,
        encoder,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        config.data.image_size,
        config.trainer.n_epochs,
        config.device,
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="default_stab")
def main(config: DictConfig):
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(config.data.image_size, int):
        config.data.image_size = (config.data.image_size, config.data.image_size)
    if isinstance(config.encoder.n_tokens, int):
        config.encoder.n_tokens = (config.encoder.n_tokens, config.encoder.n_tokens)

    model = init_model(config)
    encoder = init_encoder(config)
    train_loader, val_loader = init_dataloaders(config)
    loss_fn = init_loss(config)
    optimizer = AdamW(model.parameters(), lr=config.trainer.learning_rate)
    trainer = init_trainer(
        model, encoder, loss_fn, optimizer, train_loader, val_loader, config
    )

    trainer.launch_training(OmegaConf.to_container(config, resolve=True))


if __name__ == "__main__":
    dataset = main()  # Launch with Hydra.
