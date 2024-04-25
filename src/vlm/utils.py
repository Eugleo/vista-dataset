import json

import polars as pl
import torch as t


def serialize_dict(d) -> str:
    return json.dumps(d, sort_keys=True)


def subsample(x: t.Tensor, n_frames: int) -> t.Tensor:
    n_frames, *_ = x.shape
    step = n_frames // n_frames
    x = x[::step, ...][:n_frames, ...]
    return x


def performance_per_task(data: pl.DataFrame): ...


def confusion_matrix(data: pl.DataFrame, task: str): ...
