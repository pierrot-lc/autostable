from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler, random_split
from torchinfo import summary

from video_pretrain import VICRegEncoder, VideoDataset


def init_dataloaders(config: DictConfig) -> tuple[DataLoader, DataLoader]:
    folder_path = to_absolute_path(config.data.folder_path)
    test_size = config.data.test_size
    batch_size = config.trainer.batch_size
    n_batches = config.trainer.n_batches

    # Create datasets.
    dataset = VideoDataset.from_folder(Path(folder_path))
    train_set, test_set = random_split(dataset, lengths=[1 - test_size, test_size])

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
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        sampler=RandomSampler(
            test_set,
            replacement=True,
            num_samples=n_batches * batch_size,
        ),
        num_workers=config.trainer.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def init_model(config: DictConfig) -> VICRegEncoder:
    model = VICRegEncoder(
        config.data.image_size,
        config.data.n_channels,
        config.model.n_tokens,
        config.model.hidden_size,
        config.model.n_heads,
        config.model.ff_size,
        config.model.dropout,
        config.model.n_layers,
        config.model.projected_size,
        config.model.n_projected_layers,
    )

    # Print model summary.
    image_size = config.data.image_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    summary(
        model,
        input_size=(
            config.trainer.batch_size,
            config.data.n_channels,
            image_size[0],
            image_size[1],
        ),
        device=config.device,
    )

    return model


@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(config: DictConfig):
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = init_model(config)
    train_loader, test_loader = init_dataloaders(config)


if __name__ == "__main__":
    dataset = main()  # Launch with Hydra.
