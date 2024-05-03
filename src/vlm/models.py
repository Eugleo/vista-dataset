import base64
import hashlib
import json
import logging
import re
from functools import partial
from pathlib import Path
from typing import Callable

import backoff
import cv2
import dotenv
import openai
import polars as pl
import torch as t
from rich.progress import track
from torch.utils.data import DataLoader, IterableDataset
from torchvision.io import read_video

import vlm.utils as utils
from vlm.encoders import VideoEncoder
from vlm.heads import Head
from vlm.objects import Model, Task, Video


class VideoDataset(IterableDataset):
    def __init__(
        self, videos: list[Video], tasks: list[Task], transforms: list[Callable] = []
    ) -> None:
        self._videos = videos
        self._tasks = [task.id for task in tasks]
        self._transforms = transforms

    def __len__(self):
        return len(self._videos)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> dict:
        video = self._videos[idx]
        frames = read_video(video.path, pts_unit="sec", output_format="TCHW")[0]
        for transform in self._transforms:
            frames = transform(frames)
        return {
            "path": video.path,
            "frames": frames,
            "labels": {task: "" for task in self._tasks} | video.labels,
            "task_mask": t.tensor([task in video.labels for task in self._tasks]),
        }


class EncoderModel(Model):
    def __init__(
        self,
        id: str,
        encoder: VideoEncoder,
        heads: dict[str, list[Head]],
        batch_size: int = 1,
    ):
        self.id = id
        self._encoder = encoder
        self._heads = heads
        self._batch_size = batch_size

    def _predict_batch(self, batch, tasks: list[Task]) -> pl.DataFrame:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        videos = batch["frames"].to(device)
        video_encodings = self._encoder.encode_videos(videos)

        results = []
        for i, task in enumerate(tasks):
            mask = batch["task_mask"][:, i]

            if not mask.any():
                continue

            results += [
                {
                    "task": task.id,
                    "model": self.id,
                    "metadata": {"head": head.id, "encoder": self._encoder.metadata},
                    "video": video_path,
                    "label": label,
                    "label_idx": i,
                    "prob": prob,
                    "true_label": true_label,
                    "true_label_idx": task.labels.index(true_label),
                    "true_prob": float(true_label == label),
                }
                for head in self._heads[task.id]
                for video_path, true_label, label_probs in zip(
                    [p for i, p in enumerate(batch["path"]) if mask[i]],
                    [l for i, l in enumerate(batch["labels"][task.id]) if mask[i]],
                    head(video_encodings[mask]),
                )
                for i, (label, prob) in enumerate(zip(task.labels, label_probs))
            ]
        result = pl.DataFrame(results)

        return result

    def predict(self, videos: list[Video], tasks: list[Task]) -> pl.DataFrame:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self._encoder.to(device)

        subsample = partial(utils.subsample, n_frames=self._encoder.expected_n_frames)
        transforms = [subsample, self._encoder.transform]
        dataset = VideoDataset(videos, tasks, transforms)
        dataloader = DataLoader(dataset, batch_size=self._batch_size)

        results = []
        for batch in track(dataloader):
            results.append(self._predict_batch(batch, tasks))
        result = pl.concat(results)

        return result


class GPT4VModel(Model):
    scoring_prompt = """Now, given the original frames and your description, score the following potential video descriptions from 0 to 1 based on how well they fit the frames you've seen. Feel free to use values between 0 and 1, too. There should be exactly one 'correct' description with score 1. The descriptions are given in the following format:

- (id label) description

Options:

{classes}

The format for your answers should be:
- id label: your score
"""

    def __init__(self, n_frames: int, cache_dir: str):
        self.id = "gpt4v"
        dotenv.load_dotenv()
        self._n_frames = n_frames
        self._client = openai.OpenAI()
        self._cache_dir = cache_dir

    @staticmethod
    def _frames_to_b64(frames: t.Tensor):
        frames_np = frames.numpy()
        # Convert RGB to BGR
        frames_np = frames_np[:, :, :, ::-1]

        b64_frames = []
        for frame in frames_np:
            _, buffer = cv2.imencode(".jpg", frame)
            b64_frames.append(base64.b64encode(buffer).decode("utf-8"))  # type: ignore

        return b64_frames

    @staticmethod
    def _frame_to_payload(image):
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image}",
                "detail": "low",
                "resize": 512,
            },
        }

    @backoff.on_exception(
        backoff.expo,
        (
            openai.ConflictError,
            openai.APIStatusError,
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
            openai.UnprocessableEntityError,
            openai.APIResponseValidationError,
        ),
    )
    def _send_request(self, message, history):
        messages = history + [{"role": "user", "content": message}]
        response = self._client.chat.completions.create(
            model="gpt-4-vision-preview", messages=messages, max_tokens=1200
        )
        reponse_text = response.choices[0].message.content
        logging.info(f"Received response from GPT-4V: {reponse_text}")

        if reponse_text is None:
            raise ValueError(f"Empty response from GPT-4. {messages=}, {response=}")

        return reponse_text, [*messages, {"role": "assistant", "content": reponse_text}]

    def _predict_video(self, item: dict, task: Task, cache: dict):
        path = item["path"]
        if path not in cache:
            system_prompt = [{"type": "text", "text": task.prompt_gpt}]
            history = [{"role": "system", "content": system_prompt}]

            heading = {"type": "text", "text": "Input frames:"}
            frames = [GPT4VModel._frame_to_payload(f) for f in item["frames"]]
            _, history = self._send_request([heading, *frames], history)

            class_list = "\n".join(
                [
                    f"- ({label}) {description}"
                    for label, description in list(task.label_prompts.items())
                ]
            )
            scoring_prompt = GPT4VModel.scoring_prompt.format(classes=class_list)
            answer, history = self._send_request(scoring_prompt, history)

            label_scores = {label: 0.0 for label in task.labels} | {
                m.group(1): float(m.group(2))
                for m in re.finditer(r"- (.+): ([\d.]+)", answer)
            }
            scores = [label_scores[label] for label in task.labels]
            probs = t.Tensor(scores).softmax(0).tolist()
            label_probs = {label: prob for label, prob in zip(task.labels, probs)}

            # This writes directly into the cache object we've been passed
            cache[path] = label_probs

        return pl.DataFrame(
            [
                {
                    "task": task.id,
                    "model": self.id,
                    "metadata": {"n_frames": self._n_frames},
                    "video": path,
                    "label": label,
                    "label_idx": label_idx,
                    # All labels should be present in the cache by construction
                    "prob": cache[path][label],
                    "true_label": item["labels"][task.id],
                    "true_label_idx": task.labels.index(item["labels"][task.id]),
                    "true_prob": 1.0 if item["labels"][task.id] == label else 0.0,
                }
                for label_idx, label in enumerate(task.labels)
            ]
        )

    @staticmethod
    def _generate_cache_key(task, evaluator) -> str:
        combined = utils.serialize_dict(task) + utils.serialize_dict(evaluator)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @staticmethod
    def _load_cache(dir, task, evaluator):
        key = GPT4VModel._generate_cache_key(task, evaluator)
        filepath = Path(dir) / "score_cache" / f"{key}.json"
        if filepath.exists():
            logging.info(f"Loading cache from {filepath}")
            with open(filepath, "r") as f:
                return json.load(f)["cache"]
        else:
            logging.info(f"No cache found for {task=}, {evaluator=}")
            return {}

    @staticmethod
    def _save_cache(cache, dir, task, evaluator):
        key = GPT4VModel._generate_cache_key(task, evaluator)
        dir = Path(dir) / "score_cache"
        dir.mkdir(parents=True, exist_ok=True)
        with open(dir / f"{key}.json", "w") as f:
            logging.info(f"Saving cache to {dir / f'{key}.json'}")
            json.dump(
                {"cache": cache, "task": task, "evaluator": evaluator}, f, indent=2
            )

    def predict(self, videos: list[Video], tasks: list[Task]) -> pl.DataFrame:
        subsample = partial(utils.subsample, n_frames=self._n_frames)
        transforms = [subsample, GPT4VModel._frames_to_b64]
        dataset = list(VideoDataset(videos, tasks, transforms))

        results = []
        for item in dataset:
            for task in [task for task in tasks if task.id in item["labels"]]:
                task_info = {
                    "id": task.id,
                    "gpt4_prompt": task.prompt_gpt,
                    "labels": [
                        description for description in task.label_prompts.values()
                    ],
                }
                model_info = {"id": self.id, "n_frames": self._n_frames}
                cache = GPT4VModel._load_cache(self._cache_dir, task_info, model_info)
                results.append(self._predict_video(item, task, cache))
        result = pl.concat(results)

        return result
