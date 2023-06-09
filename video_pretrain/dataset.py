import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import VideoReader


class VideoDataset(Dataset):
    def __init__(self, video_paths: list[Path]):
        """Simple video dataset that loads a pair of images
        coming from the same video.

        ---
        Args:
            video_paths: List of video paths, from which
                frames will be sampled.
        """
        super().__init__()
        self.video_paths = video_paths

    def __len__(self) -> int:
        """Get the number of videos in the dataset.

        ---
        Returns:
            Number of videos in the dataset.
        """
        return len(self.video_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get a pair of images from the same video.

        ---
        Args:
            index: Index of the video to pick the images from.

        ---
        Returns:
            A pair of images from the same video.
                Shape of [2, n_channels, height, width].
        """
        # Sample with replacement the images from the same video.
        video_path = self.video_paths[index]
        reader = VideoReader(str(video_path), stream="video")

        # Sample two frames from the video.
        total_duration = reader.get_metadata()["video"]["duration"][0]
        durations = [random.uniform(0, total_duration) for _ in range(2)]
        frames = [
            next(reader.seek(frame_duration))["data"] for frame_duration in durations
        ]

        return torch.stack(frames)

    @classmethod
    def from_folder(cls, folder_path: Path) -> "VideoDataset":
        """Read all videos from a folder and construct the dataset."""
        video_paths = [
            path
            for path in folder_path.iterdir()
            if path.suffix in [".mp4", ".avi", ".mov"]
        ]
        return cls(video_paths)
