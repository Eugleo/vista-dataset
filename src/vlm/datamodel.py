from typing import Callable

import torch as t
from pydantic.dataclasses import dataclass
from torch.utils.data import IterableDataset
from torchvision.io import read_video


@dataclass
class Task:
    id: str
    prompt_gpt: str
    label_prompts: dict[str, str]
    prompt_baseline: str

    @property
    def labels(self) -> list[str]:
        return list(self.label_prompts.keys())

    @staticmethod
    def from_dict(val):
        return Task(val["id"], val["gpt4_prompt"], val["baseline"], val["labels"])


@dataclass
class Video:
    path: str
    labels: dict[str, str]


class VideoDataset(IterableDataset):
    def __init__(
        self, videos: list[Video], tasks: list[Task], transforms: list[Callable] = []
    ) -> None:
        self._videos = videos
        self._tasks = [task.id for task in tasks]
        self._transforms = transforms

    def __len__(self):
        return len(self._videos)

    def __getitem__(self, idx: int) -> dict:
        video = self._videos[idx]
        frames = read_video(video.path, pts_unit="sec", output_format="TCHW")[0]
        for transform in self._transforms:
            frames = transform(frames)
        return {
            "path": video.path,
            "frames": frames,
            "labels": video.labels,
            "task_mask": t.tensor([task in video.labels for task in self._tasks]),
        }
