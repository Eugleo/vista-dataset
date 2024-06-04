import logging
import math
import re
import uuid
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import backoff
import dotenv
import jsonlines
import openai
import polars as pl
import torch as t
from rich.progress import Progress, track
from torch.utils.data import DataLoader, IterableDataset
import torchvision
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
        logging.info(f"VideoDataset initialized with {len(self)} videos")

    def __len__(self):
        return len(self._videos)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> dict:
        video = self._videos[idx]
        frames = read_video(video.path, pts_unit="sec", output_format="TCHW")[0]
        for transform in self._transforms:
            try:
                frames = transform(frames)
            except Exception as e:
                raise ValueError(f"Error transforming video {video.path}: {e}")
        return {
            "path": video.path,
            "frames": frames,
            "labels": {task: "" for task in self._tasks} | video.labels,
            "task_mask": t.tensor([task in video.labels for task in self._tasks]),
        }


class EncoderModel(Model):
    def __init__(
        self,
        encoder: VideoEncoder,
        heads: dict[str, list[Head]],
        batch_size: int = 1,
    ):
        self.id = encoder.id
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
                    "model": self._encoder.id + "_" + head.id,
                    "metadata": {"head": head.id, "encoder": self._encoder.metadata},
                    "video": video_path,
                    "label": label,
                    "label_idx": i,
                    "score": score,
                    "true_label": true_label,
                    "true_label_idx": ",".join(str(task.labels.index(label)) for label in true_label.split(",")),
                    # in multilabel regime, joins correct indices separated by comma,
                    # in multiclass regime keeps a single index (converted to str type)
                }
                for head in self._heads[task.id]
                for video_path, true_label, label_scores in zip(
                    [p for i, p in enumerate(batch["path"]) if mask[i]],
                    [l for i, l in enumerate(batch["labels"][task.id]) if mask[i]],
                    head(video_encodings[mask]),
                )
                for i, (label, score) in enumerate(zip(task.labels, label_scores))
            ]
        result = pl.DataFrame(results)

        return result

    def predict(
        self, videos: list[Video], tasks: list[Task], _log_dir: Path
    ) -> pl.DataFrame:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self._encoder.to(device)

        subsample = partial(utils.subsample, n_frames=self._encoder.expected_n_frames)
        transforms = [subsample, self._encoder.transform]
        dataset = VideoDataset(videos, tasks, transforms)
        dataloader = DataLoader(dataset, batch_size=self._batch_size)

        results = []
        # for batch in track(dataloader):
        # force the dataloader to load all the videos up front, so that if any are invalid, an error will be thrown before any predictions are made
        for batch in track(list(dataloader)):
            results.append(self._predict_batch(batch, tasks))
        result = pl.concat(results)

        return result



MULTICLASS_PROMPT = """Then, given the original frames and your description, score the following potential video descriptions from 0 to 1 based on how well they describe the video you've seen. Feel free to use values between 0 and 1, too. There should be exactly one 'correct' description with score 1. The descriptions are given in the following format:

- (label) description

Options:

{classes}

The final scores should be in the following format, verbatim:

```
- (label) your score
```

Be sure not to alter the label in any way, since we will use it to match your scores to the potential descriptions we've given you.
"""

MULTILABEL_PROMPT = """Then, given the original frames and your description, score the following potential video descriptions from 0 to 1 based on how well they describe the video you've seen. Feel free to use values between 0 and 1, too. There could be more than one 'correct' description with score 1. The descriptions are given in the following format:

- (label) description

Options:

{classes}

The final scores should be in the following format, verbatim:

```
- (label) your score
```

Be sure not to alter the label in any way, since we will use it to match your scores to the potential descriptions we've given you.
"""


class GPTModel(Model):
    def __init__(self, n_frames: int, cache_dir: str, async_batch: bool, model: str, task_mode: str):
        self.id = model
        dotenv.load_dotenv()
        self._n_frames = n_frames
        self._client = openai.OpenAI()
        self._cache_dir = cache_dir
        self._model = model
        self._task_mode = task_mode
        self._async_batch = async_batch

        if task_mode == "multiclass":
            self.scoring_prompt = MULTICLASS_PROMPT
        elif task_mode == "multilabel":
            self.scoring_prompt = MULTILABEL_PROMPT
        else:
            raise ValueError(f"Unknown task mode {task_mode}")

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
    def _send_request(self, history):
        response = self._client.chat.completions.create(
            model=self._model, messages=history, max_tokens=2500
        )
        reponse_text = response.choices[0].message.content  # type: ignore
        return reponse_text or "", history + [
            {"role": "assistant", "content": reponse_text}
        ]

    @staticmethod
    def parse_and_cache_scores(
        response: str,
        path: str,
        task_labels: list[str],
        cache: dict,
        normalize: bool = False,
        verbose: bool = False,
    ):
        response_scores = {}

        for l in task_labels:
            relevant_lines = [
                line
                for line in response.splitlines()
                if re.findall(rf"\({l}\)", line, flags=re.IGNORECASE)
            ]

            scores = []
            for line in relevant_lines:
                scores += re.findall(r"(?<!_)([0-9]+(?:\.[0-9]+)?)", line)
            if not scores:
                continue

            response_scores[l] = max(float(s) for s in scores)

        # for m in re.finditer(
        #     r"-[^a-zA-Z\d]*([a-zA-Z_\d-]+)[^a-zA-Z\d]*:[^a-zA-Z\d]*([\d.]*\d)",
        #     response,
        # ):
        #     label = m.group(1)
        #     score = float(m.group(2))
        #     response_scores[label] = score

        if verbose and any(
            (l not in response_scores and l in response)
            for l in [label for label in task_labels if label in path.split("/")]
        ):
            print(f"WARNING: Missing scores for some labels in {path}", flush=True)
            print(f"Response: {response}", flush=True)
            print("", flush=True)

        label_scores = {label: -0.1 for label in task_labels} | response_scores
        scores = [label_scores[label] for label in task_labels]
        if normalize:
            scores = t.Tensor(scores).softmax(0).tolist()
        label_probs = {label: prob for label, prob in zip(task_labels, scores)}

        # This writes directly into the cache object we've been passed
        cache[path] = label_probs

        return label_probs

    def _predict_video(
        self, item: dict, task: Task, cache: dict, log_writer: jsonlines.Writer
    ):
        path = item["path"]

        logging.info(f"Predicting task {task.id} for video: {path}")
        logging.info(f"True label: {item['labels'][task.id]}")
        if path in cache:
            logging.info(f"Using cached scores: {cache[path]}")
        else:
            logging.info("No cached scores found")
            history = self._build_batch_message(item, task)
            answer, history = self._send_request(history)

            GPTModel.parse_and_cache_scores(answer, path, task.labels, cache)

            log_writer.write(
                {
                    "video": path,
                    "task": task.id,
                    "model": self.id,
                    "history": history,
                    "parsed_scores": {
                        label: cache[path][label] for label in task.labels
                    },
                    "predicted_label": max(cache[path], key=cache[path].get),
                    "true_label": item["labels"][task.id],
                    "label_descriptions": task.label_prompts,
                }
            )

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
                    "score": cache[path][label],
                    "true_label": item["labels"][task.id],
                    "true_label_idx": task.labels.index(item["labels"][task.id]) if self._task_mode == "multiclass" \
                        else ",".join(str(task.labels.index(label)) for label in item["labels"][task.id].split(",")),    # multi-label regime
                }
                for label_idx, label in enumerate(task.labels)
            ]
        )

    def _build_batch_message(self, item: dict, task: Task):
        assert task.prompt_gpt is not None

        class_list = "\n".join(
            [
                f"- ({label}) {description}"
                for label, description in list(task.label_prompts.items())
            ]
        )
        scoring_prompt = self.scoring_prompt.format(classes=class_list)

        heading = {"type": "text", "text": "Input frames:"}
        frames = [GPTModel._frame_to_payload(f) for f in item["frames"]]

        history = [
            {"role": "system", "content": task.prompt_gpt + scoring_prompt},
            {"role": "user", "content": [heading, *frames]},
        ]

        return history

    def _build_async_message(
        self, item: dict, task: Task, log_writer: jsonlines.Writer
    ):
        video = item["path"]
        history = self._build_batch_message(item, task)
        log_writer.write(
            {
                "video": video,
                "task": task.id,
                "model": self.id,
                "history": history,
                "true_label": item["labels"][task.id],
                "label_descriptions": task.label_prompts,
            }
        )

        return {
            "custom_id": f"{video},{task.id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self._model,
                "messages": history,
                "max_tokens": 1200,
            },
        }

    def _build_async_requests(self, dataset, tasks, group, log_writer):
        requests = []
        with Progress() as progress:
            processing_task = progress.add_task(
                "Predicting...", total=len(dataset) * len(tasks)
            )

            for item in dataset:
                for task in tasks:
                    progress.update(processing_task, advance=1)
                    if task.id.startswith(group) and item["labels"][task.id]:
                        requests.append(
                            self._build_async_message(item, task, log_writer)
                        )
        return requests

    @staticmethod
    def _log_video(cache_dir, item):
        video_logging_dir = Path(cache_dir) / "gpt_input_images"
        video_logging_dir.mkdir(parents=True, exist_ok=True)

        frames = utils.b64_to_frames(item["frames"])
        frames = torchvision.transforms.Resize(size=(512, 512))(frames)
        video_as_grid = torchvision.utils.make_grid(frames, nrow=math.ceil(math.sqrt(frames.shape[0])), normalize=True)
        video_as_grid = torchvision.transforms.functional.to_pil_image(video_as_grid)
        video_as_grid.save(video_logging_dir / (Path(item["path"]).stem + ".png"))

    def predict(
        self, videos: list[Video], tasks: list[Task], log_dir: Path
    ) -> Optional[pl.DataFrame]:
        logging.info("Configuring dataset...")
        subsample = partial(utils.subsample, n_frames=self._n_frames)
        transforms = [subsample, utils.frames_to_b64]
        dataset_iter = VideoDataset(videos, tasks, transforms)
        # converting to a list forces all the videos to be converted up front, so that if any are invalid, an error will be thrown before any GPT-4 calls are made
        logging.info("Processing videos...")
        dataset = list(dataset_iter)

        for item in dataset:
            GPTModel._log_video(self._cache_dir, item)

        logging.info("Predicting...")

        log_dir = (
            log_dir / self.id
            if not self._async_batch
            else log_dir / f"requests_{self.id}"
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        with jsonlines.open(
            log_dir / f"{uuid.uuid4()}.jsonl", "w", sort_keys=True
        ) as log_writer:
            if self._async_batch:
                print("WARNING: Ignoring all cache when using async_batch", flush=True)

                batch_ids = []

                for group in set("/".join(task.id.split("/")[:2]) for task in tasks):
                    task_list = self._build_async_requests(
                        dataset, tasks, group, log_writer
                    )

                    with jsonlines.open(".cache/batchinput.jsonl", "w") as writer:
                        writer.write_all(task_list)

                    batch_input_file = self._client.files.create(
                        file=open(".cache/batchinput.jsonl", "rb"), purpose="batch"
                    )

                    batch_object = self._client.batches.create(
                        input_file_id=batch_input_file.id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                    )

                    batch_ids.append(batch_object.id)

                logging.info(f"Created batches: {batch_ids}")

                return None

            else:
                print("WARNING: Evaluating synchronously", flush=True)
                results = []

                with Progress() as progress:
                    processing_task = progress.add_task(
                        "Predicting...", total=len(dataset) * len(tasks)
                    )
                    for item in dataset:
                        for task in tasks:
                            progress.update(processing_task, advance=1)
                            if not item["labels"][task.id]:
                                continue
                            task_info = {
                                "prompt_gpt": task.prompt_gpt,
                                "labels": [
                                    description
                                    for description in task.label_prompts.values()
                                ],
                            }
                            model_info = {"id": self.id, "n_frames": self._n_frames}
                            cache = utils.load_cache(
                                self._cache_dir, task_info, model_info
                            )
                            results.append(
                                self._predict_video(item, task, cache, log_writer)
                            )
                            utils.save_cache(
                                cache, self._cache_dir, task_info, model_info
                            )
                result = pl.concat(results)

                return result
