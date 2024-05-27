import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

import cv2
import einops
import numpy as np
import plotly.express as px
import polars as pl
import sklearn.metrics as skm
import torch as t
from polars import col as c


def get_experiment_dir(experiment_dir, experiment_id):
    if experiment_id is None:
        experiments = [d for d in Path(experiment_dir).iterdir() if d.is_dir()]
        dir = sorted(experiments, key=lambda d: d.stat().st_mtime)[-1]
        print(f"Loading the most recent experiment: {dir}...")
        return dir
    else:
        return Path(experiment_dir) / experiment_id


def serialize_dict(d) -> str:
    return json.dumps(d, sort_keys=True)


def subsample(x: t.Tensor, n_frames: int) -> t.Tensor:
    total_frames, *_ = x.shape
    if total_frames <= n_frames:
        raise ValueError("Video is too short to subsample.")

    if (total_frames - n_frames) % (n_frames - 1) != 0:
        # Replicate the last frame to make sure it will be selected
        last_frame = einops.repeat(
            x[-1], "... -> n ...", n=(n_frames - (total_frames % (n_frames - 1)))
        )
        x = t.cat([x, last_frame])
        total_frames, *_ = x.shape
        assert (total_frames - n_frames) % (n_frames - 1) == 0
    step = (total_frames - n_frames) // (n_frames - 1) + 1
    x_subsampled = x[::step]
    assert len(x_subsampled) == n_frames
    assert (x[0] == x_subsampled[0]).all() and (x[-1] == x_subsampled[-1]).all()
    return x_subsampled


def rescale(df: pl.DataFrame, in_each: Literal["label", "video"]) -> pl.DataFrame:
    if in_each == "video":
        df = df.with_columns(
            (c("score").exp() / c("score").exp().sum())
            .over("model", "task", "video")
            .alias("score")
        )
        # scores_stats = df.groupby(["model", "task", "video"]).agg(
        #     c("score").mean().alias("mean_score"), c("score").std().alias("std_score")
        # )
        # df = df.join(scores_stats, on=["model", "task", "video"]).with_columns(
        #     score=(c("score") - c("mean_score")) / (c("std_score") + 1e-6)
        # )
    elif in_each == "label":
        scores_stats = df.groupby(["model", "task", "label"]).agg(
            c("score").mean().alias("mean_score"), c("score").std().alias("std_score")
        )
        df = df.join(scores_stats, on=["model", "task", "label"]).with_columns(
            score=pl.when(~c("model").str.contains("gpt"))
            .then((c("score") - c("mean_score")) / (c("std_score") + 1e-6))
            .otherwise(c("score"))
        )
    return df


def get_predictions(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by(["video", "task", "model"]).agg(
        # Extract the label with the highest probability
        pl.col("label").sort_by("score").last(),
        pl.col("true_label").first(),
    )


# A helper function used to extract label colums from the dataframe,
# package them as structs, and then map matric_fun over them
def compute_metric(metric_fun):
    return pl.struct("true_label", "label").map_batches(metric_fun).first()


def performance_per_task(data: pl.DataFrame):
    return px.bar(
        data.to_pandas(),
        x="model",
        color="model",
        facet_col="task",
        y="mcc",
        title="Performance of each model per task",
    )


def add_random_baseline(scores: pl.DataFrame):
    random_model_scores = scores.unique(["video", "task", "label"])
    random_model_scores = random_model_scores.with_columns(
        score=pl.lit(np.random.rand(len(random_model_scores))),
        model=pl.lit("ω random"),
    )
    return pl.concat([scores, random_model_scores])


def accuracy(group: pl.Series):
    return skm.accuracy_score(
        y_true=group.struct.field("true_label").to_numpy(),
        y_pred=group.struct.field("label").to_numpy(),
    )


def _generate_cache_key(task, evaluator) -> str:
    combined = serialize_dict(task) + serialize_dict(evaluator)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def load_cache(dir, task, model):
    key = _generate_cache_key(task, model)
    filepath = Path(dir) / "scores" / f"{key}.json"
    if filepath.exists():
        logging.info(f"Loading cache from {filepath}")
        with open(filepath, "r") as f:
            return json.load(f)["cache"]
    else:
        logging.info(f"No cache found for {task=}, {model=}")
        return {}


def save_cache(cache, dir, task, model):
    key = _generate_cache_key(task, model)
    dir = Path(dir) / "scores"
    dir.mkdir(parents=True, exist_ok=True)
    with open(dir / f"{key}.json", "w") as f:
        logging.info(f"Saving cache to {dir / f'{key}.json'}")
        json.dump({"cache": cache, "task": task, "model": model}, f, indent=2)


def frames_to_b64(frames: t.Tensor):
    frames = einops.rearrange(frames, "t c h w -> t h w c")
    frames_np = frames.numpy()
    # Convert RGB to BGR
    frames_np = frames_np[:, :, :, ::-1]

    b64_frames = []
    for frame in frames_np:
        _, buffer = cv2.imencode(".jpg", frame)
        b64_frames.append(base64.b64encode(buffer).decode("utf-8"))  # type: ignore

    return b64_frames
