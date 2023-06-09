import hydra
from omegaconf import DictConfig, OmegaConf

from video_pretrain import UViT


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
    print(model)


if __name__ == "__main__":
    main()  # Launch with Hydra.
