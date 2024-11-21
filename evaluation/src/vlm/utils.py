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
from scipy.stats import hypergeom


def get_baseline_ap(num_documents: int, num_relevant: int) -> float:
    ap = 0
    for i in range(1, num_relevant + 1):
        for n in range(i, num_documents - num_relevant + i + 1):
            ap += hypergeom.pmf(i, num_documents, num_relevant, n) * (i / n) ** 2
    ap = ap / num_relevant
    assert isinstance(ap, float)
    return ap


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


def add_rescaling(scores: pl.DataFrame) -> pl.DataFrame:
    # lab = _rescale_in_label(scores)
    vid_lab = _rescale_in_label(_rescale_in_video(scores))
    return pl.concat(
        [
            # scores.with_columns(rescaling=pl.lit("n")),
            # lab.with_columns(rescaling=pl.lit("l")),
            vid_lab.with_columns(rescaling=pl.lit("v+l")),
        ]
    )


def _rescale_in_video(scores: pl.DataFrame) -> pl.DataFrame:
    return scores.with_columns(
        (c("score").exp() / c("score").exp().sum())
        .over("model", "task", "video")
        .alias("score")
    )


def _rescale_in_label(scores: pl.DataFrame) -> pl.DataFrame:
    # Calculate mean and std directly within groups, avoiding the expensive self-join
    stats = scores.group_by(["model", "task", "label"]).agg(
        [
            c("score").mean().alias("score_mean"),
            c("score").std().alias("score_std"),
        ]
    )

    # Single join with the stats and compute z-score
    return (
        scores.join(stats, on=["model", "task", "label"])
        .with_columns(
            ((c("score") - c("score_mean")) / (c("score_std") + 1e-6)).alias("score")
        )
        .drop("score_mean", "score_std")
    )


def get_predictions(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by(["video", "task", "model"]).agg(
        # Extract the label with the highest probability
        c("label").sort_by("score").last(),
        c("true_label").first(),
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


def add_majority_baseline(predictions: pl.DataFrame):
    majority_label = predictions.group_by(["task"]).agg(
        label=c("true_label").mode().first()
    )
    majority_baseline = (
        predictions.select("task", "video", "true_label")
        .unique()
        .join(majority_label, on=["task"])
        .with_columns(model=pl.lit("majority_baseline"))
    ).select(*predictions.columns)
    return pl.concat([predictions, majority_baseline])


def accuracy(group: pl.Series):
    return skm.accuracy_score(
        y_true=group.struct.field("true_label").to_numpy(),
        y_pred=group.struct.field("label").to_numpy(),
    )


def f1(group: pl.Series):
    return skm.f1_score(
        y_true=group.struct.field("true_label").to_numpy(),
        y_pred=group.struct.field("label").to_numpy(),
        average="macro",
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
