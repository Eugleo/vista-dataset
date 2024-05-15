import json
import time
from pathlib import Path
from typing import Callable, Literal, Optional

import torch as t
import yaml
from human_id import generate_id
from pydantic import BaseModel, ValidationInfo, field_validator, model_validator

from vlm.encoders import CLIP, S3D, TextEncoder, ViCLIP
from vlm.heads import CosineHead, Head
from vlm.models import EncoderModel, GPT4VModel, Model
from vlm.objects import Experiment, Task, Video


class HeadConfig(BaseModel):
    kind: Literal["cosine"]
    alpha: Optional[float] = None

    @field_validator("alpha", mode="after")
    def alpha_present(cls, v: Optional[float], info: ValidationInfo):
        kind = info.data["kind"]
        if kind == "projection" and v is None:
            raise ValueError("Projection head requires an alpha value")
        if kind != "projection" and v is not None:
            raise ValueError("Alpha should only be present for a projection head")
        return v

    def to_head(self, task: Task, encoder: TextEncoder) -> Head:
        # if self.kind == "projection":
        #     assert self.alpha is not None
        #     return ProjectionHead.for_task(task, encoder, self.alpha)
        device = "cuda" if t.cuda.is_available() else "cpu"
        encoder.to(device)
        if self.kind == "cosine":
            return CosineHead.for_task(task, encoder)
        else:
            raise ValueError(f"Unknown head: {self.kind}")


class ModelConfig(BaseModel):
    kind: Literal["encoder", "gpt"]
    batch_size: Optional[int] = None
    encoder: Optional[Literal["clip", "s3d", "viclip"]] = None
    heads: Optional[list[HeadConfig]] = None
    n_frames: Optional[int] = None
    task_mode: Optional[str] = None
    hf_model: Optional[str] = None

    @model_validator(mode="before")
    def encoder_valid(cls, data: dict):
        if data["kind"] == "encoder":
            if (
                "encoder" not in data
                or "heads" not in data
                or not isinstance(data["heads"], list)
                or len(data["heads"]) == 0
            ):
                raise ValueError(
                    "Model of kind 'encoder' requires an encoder and at least one head"
                )
            if "batch_size" not in data:
                data["batch_size"] = 8
            if data["encoder"] == "clip":
                if "n_frames" not in data:
                    raise ValueError("A CLIP encoder requires the number of frames")
                if "hf_model" not in data or len(data["hf_model"].split("/")) != 2:
                    raise ValueError(
                        "CLIP encoder requires a HuggingFace model in the repo/model format"
                    )
            elif "n_frames" in data or "hf_model" in data:
                raise ValueError(
                    "Number of frames and HuggingFace model should only be present for a CLIP encoder"
                )

        return data

    @model_validator(mode="before")
    def gpt_valid(cls, data: dict):
        if data["kind"] == "gpt":
            if "n_frames" not in data:
                raise ValueError("A GPT model requires the number of frames")
            if "batch_size" in data:
                raise ValueError("A GPT model should not have a batch size")
            if "task_mode" not in data:
                raise ValueError("A GPT model needs a clarification on task mode (multilabel/multiclass -- this influences system prompt)")
        return data

    @model_validator(mode="before")
    def non_encoder_valid(cls, data: dict):
        if data["kind"] != "encoder":
            if "encoder" in data:
                raise ValueError("Only models of kind 'encoder' require an encoder")
            if "heads" in data:
                raise ValueError("Only models of kind 'encoder' require heads")
            if "hf_model" in data:
                raise ValueError("Only CLIP encoders require a HuggingFace model")
        return data

    def _get_encoder(self, cache_dir: str):
        encoder_cache_dir = str(Path(cache_dir) / "encoders")
        if self.encoder == "clip":
            assert self.hf_model is not None and self.n_frames is not None
            model_name, pretrained = self.hf_model.split("/")
            return CLIP(model_name, pretrained, encoder_cache_dir, self.n_frames)
        elif self.encoder == "s3d":
            return S3D(encoder_cache_dir)
        elif self.encoder == "viclip":
            return ViCLIP(encoder_cache_dir)
        else:
            raise ValueError(f"Unknown encoder: {self.encoder}")

    def _get_heads(
        self, tasks: list[Task], encoder: TextEncoder
    ) -> dict[str, list[Head]]:
        heads = {}
        assert self.heads is not None
        for task in tasks:
            heads[task.id] = [head.to_head(task, encoder) for head in self.heads]
        return heads

    def to_model(self, tasks: list[Task], cache_dir: str) -> Callable[[], Model]:
        if self.kind == "encoder":

            def get_encoder():  # type: ignore
                assert self.encoder is not None
                assert self.batch_size is not None
                encoder = self._get_encoder(cache_dir)
                heads = self._get_heads(tasks, encoder)
                return EncoderModel(
                    encoder=encoder, heads=heads, batch_size=self.batch_size
                )

            return get_encoder

        elif self.kind == "gpt":

            def get_gpt():
                assert self.n_frames is not None
                return GPT4VModel(n_frames=self.n_frames, cache_dir=cache_dir, task_mode=self.task_mode)

            return get_gpt

        else:
            raise ValueError(f"Unknown model kind: {self.kind}")


class ExperimentConfig(BaseModel):
    config_file: str
    tasks: list[str]
    models: list[ModelConfig]

    video_dir: str
    task_dir: str
    cache_dir: str = "/data/datasets/vlm_benchmark/.cache"
    output_dir: str = "/data/datasets/vlm_benchmark/experiments"

    @field_validator("video_dir", mode="after")
    def video_dir_exists(cls, v: str, _: ValidationInfo):
        if not Path(v).is_dir():
            raise ValueError(f"Video directory '{v}' does not exist")
        return v

    @field_validator("task_dir", mode="after")
    def task_dir_exists(cls, v: str, _: ValidationInfo):
        if not Path(v).is_dir():
            raise ValueError(f"Task directory '{v}' does not exist")
        return v

    @field_validator("cache_dir", mode="after")
    def cache_dir_exists(cls, v: str, _: ValidationInfo):
        if not Path(v).is_dir():
            raise ValueError(f"Cache directory '{v}' does not exist")
        return v

    def _load_videos(self, tasks: list[Task]) -> list[Video]:
        videos = {}
        for task in tasks:
            data_path = Path(self.task_dir) / f"{task.id}_data.json"
            with open(data_path) as f:
                items = json.load(f)
            for item in items:
                labels = videos.setdefault(item["path"], {})
                if task.id in labels:
                    raise ValueError(f"Duplicate label for video {item['path']}")
                if item["label"] not in task.labels:
                    raise ValueError(
                        f"Invalid label '{item['label']}' for task '{task.id}'"
                    )
                labels[task.id] = item["label"]
        # TODO: Change this to a use Path
        return [
            Video(f"{self.video_dir}/{path}", labels) for path, labels in videos.items()
        ]

    def _load_tasks(self) -> list[Task]:
        tasks = []
        for task_id in self.tasks:
            task_path = Path(self.task_dir) / f"{task_id}.yaml"
            task = Task.from_file(task_id, task_path)
            has_gpt = any(model.kind == "gpt" for model in self.models)
            if has_gpt and task.prompt_gpt is None:
                raise ValueError(f"Task {task.id} requires a GPT prompt")
            has_projection = any(
                model.heads is not None
                and any(head.kind == "projection" for head in model.heads)
                for model in self.models
            )
            if has_projection and task.prompt_baseline is None:
                raise ValueError(f"Task {task.id} requires a baseline prompt")

            tasks.append(task)

        return tasks

    @staticmethod
    def from_file(path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return ExperimentConfig(config_file=path, **data)

    def to_experiment(self) -> Experiment:
        id = f"Exp_{time.strftime('%m%d%H%M%S')}_{generate_id(word_count=3)}"
        tasks = self._load_tasks()
        videos = self._load_videos(tasks)
        models = [model.to_model(tasks, self.cache_dir) for model in self.models]
        return Experiment(id, tasks, videos, models, self.config_file, self.output_dir)
