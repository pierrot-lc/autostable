from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler, random_split

from video_pretrain import UViT, VideoDataset


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


def init_model(config: DictConfig) -> UViT:
    model = UViT(
        config.data.image_size,
        config.data.n_channels,
        config.model.n_tokens,
        config.model.hidden_size,
        config.model.n_heads,
        config.model.ff_size,
        config.model.dropout,
        config.model.n_layers,
    )
    return model


@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(config: DictConfig):
    model = init_model(config)
    train_loader, test_loader = init_dataloaders(config)


if __name__ == "__main__":
    dataset = main()  # Launch with Hydra.
