"""Video dataset that returns a sequence of consecutive frames.
"""
import random
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.io import VideoReader
from torchvision.transforms.functional import resize

torchvision.set_video_backend("pyav")


class ContinuousDataset(Dataset):
    def __init__(self, video_paths: list[Path], image_size: list[int], n_frames: int):
        """Video dataset that loads a sequence of consecutive frames
        coming from the same video.

        ---
        Args:
            video_paths: List of video paths, from which
                frames will be sampled.
            image_size: Size of the images to return.
                List of [height, width].
            n_frames: Number of consecutive frames to sample from each video.
        """
        super().__init__()
        self.video_paths = video_paths
        self.image_size = image_size
        self.n_frames = n_frames

    def __len__(self) -> int:
        """Get the number of videos in the dataset.

        ---
        Returns:
            Number of videos in the dataset.
        """
        return len(self.video_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Sample a sequence of consecutive frames from the same video.

        ---
        Args:
            index: Index of the video to pick the images from.

        ---
        Returns:
            A sequence of consecutive frames from the same video.
                Shape of [n_frames, n_channels, height, width].
        """
        video_path = self.video_paths[index]
        reader = VideoReader(str(video_path), stream="video")

        # Determine the number of frames in the video.
        metadata = reader.get_metadata()["video"]
        total_duration = metadata["duration"][0]
        frame_rate = metadata["fps"][0]
        total_frames = int(total_duration * frame_rate)

        # Sample a sequence of consecutive frames from the video.
        start_frame = random.randint(0, total_frames - self.n_frames)
        reader.seek(
            start_frame / frame_rate, keyframes_only=True
        )  # Jump to the start frame.
        frames = [next(reader)["data"] for _ in range(self.n_frames)]
        frames = [resize(frame, self.image_size, antialias=True) for frame in frames]
        frames = [frame / 255 for frame in frames]

        return torch.stack(frames)

    @classmethod
    def from_folder(
        cls, folder_path: Path, image_size: tuple[int, int] | list[int], n_frames: int
    ) -> "ContinuousDataset":
        """Read all videos from a folder and construct the dataset."""
        video_paths = [
            path
            for path in folder_path.iterdir()
            if path.suffix in [".mp4", ".avi", ".mov"]
        ]
        return cls(video_paths, list(image_size), n_frames)
