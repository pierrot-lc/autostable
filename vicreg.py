from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, RandomSampler, random_split
from torchinfo import summary
from vicreg_loss import VICRegLLoss

from autostable import SampleDataset, VICRegEncoder, VICRegTrainer


def init_dataloaders(config: DictConfig) -> tuple[DataLoader, DataLoader]:
    folder_path = to_absolute_path(config.data.folder_path)
    test_size = config.data.test_size
    batch_size = config.trainer.batch_size
    n_batches = config.trainer.n_batches

    # Create datasets.
    dataset = SampleDataset.from_folder(Path(folder_path), config.data.image_size)
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


def init_model(config: DictConfig) -> VICRegEncoder:
    model = VICRegEncoder(
        config.data.image_size,
        config.data.n_channels,
        config.encoder.n_tokens,
        config.encoder.hidden_size,
        config.encoder.n_layers,
        config.encoder.projected_size,
        config.encoder.n_projected_layers,
    )

    # Print model summary.
    summary(
        model,
        input_size=(
            config.trainer.batch_size,
            config.data.n_channels,
            config.data.image_size[0],
            config.data.image_size[1],
        ),
        device=config.device,
    )

    return model


def init_loss(config: DictConfig) -> VICRegLLoss:
    return VICRegLLoss(
        config.vicreg.num_matches,
        config.vicreg.alpha,
        config.vicreg.inv_coeff,
        config.vicreg.var_coeff,
        config.vicreg.cov_coeff,
        config.vicreg.gamma,
    )


def init_trainer(
    model: VICRegEncoder,
    loss_fn: VICRegLLoss,
    optimizer: Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: DictConfig,
) -> VICRegTrainer:
    return VICRegTrainer(
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        config.trainer.n_epochs,
        config.device,
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="default_vicreg")
def main(config: DictConfig):
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(config.data.image_size, int):
        config.data.image_size = (config.data.image_size, config.data.image_size)
    if isinstance(config.encoder.n_tokens, int):
        config.encoder.n_tokens = (config.encoder.n_tokens, config.encoder.n_tokens)

    model = init_model(config)
    train_loader, val_loader = init_dataloaders(config)
    loss_fn = init_loss(config)
    optimizer = AdamW(model.parameters(), lr=config.trainer.learning_rate)
    trainer = init_trainer(model, loss_fn, optimizer, train_loader, val_loader, config)

    trainer.launch_training(config)


if __name__ == "__main__":
    dataset = main()  # Launch with Hydra.
