import logging
import re
from functools import partial
from typing import Callable, Optional

import backoff
import dotenv
import jsonlines
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
                    "true_label_idx": task.labels.index(true_label),
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

    def predict(self, videos: list[Video], tasks: list[Task]) -> pl.DataFrame:
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


class GPTModel(Model):
    scoring_prompt = """Now, given the original frames and your description, score the following potential video descriptions from 0 to 1 based on how well they describe the video you've seen.

NOTE 1: If none of the descriptions seem to match, e.g. if they mention an object you have not described above, you should try to reinterpret the frames, or your description of them. Probably they mean something different than you originally thought. In any case, we are SURE there is exactly one correct description that should be given a score of 1.

NOTE 2: Some details in the descriptions might be incorrect, or might have been unseen by you since you only got 5 frames from the video. The important things are the actions, the objects they DIRECTLY involve, and the order of the actions, nothing else. For example, feel free to ignore details such as "we turn left" and "to the right of a laptop" and similar if they are not relevant to the actions. Pay VERY good attention to the order.

The descriptions are given in the following format:

- (id label) description

Options:

{classes}

The format for your answers should be:
- id label: your score

But first, include some step-by-step thinking on the matter.
"""

    scoring_prompt_v2 = """

# SECOND TASK

Your second task, only after you finish describing the frames, will be — given the original frames and your description — to score the following potential video descriptions from 0 to 100 based on how well they describe the video you've seen.

NOTE 1: If none of the descriptions seem to match, e.g. if they mention an object you have not described above, you should try to reinterpret the frames, or your description of them. Probably they mean something different than you originally thought. In any case, we are SURE there is exactly one correct description that should be given a score of 1.

NOTE 2: Some details in the descriptions might be incorrect, or might have been unseen by you since you only got 5 frames from the video. The important things are the actions, the objects they DIRECTLY involve, and the order of the actions, nothing else. For example, feel free to ignore details such as "we turn left" and "to the right of a laptop" and similar if they are not relevant to the actions. Pay VERY good attention to the order.

Options, in the format `- (id label) description`:

{classes}

The format for your scores should be:
- id label: your score

Write your answer in three sections: frame descriptions, discussion of a small selection of the most likely labels, final scores in the correct format. In the scores section, start with the most likely labels to make sure you fit into the token limit.
"""

    def __init__(self, n_frames: int, cache_dir: str, async_batch: bool, model: str):
        self.id = model
        dotenv.load_dotenv()
        self._n_frames = n_frames
        self._client = openai.OpenAI()
        self._cache_dir = cache_dir
        self._model = model
        self._async_batch = async_batch

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
            model=self._model, messages=messages, max_tokens=1200
        )
        reponse_text = response.choices[0].message.content  # type: ignore
        logging.info(f"GPT response: {reponse_text}")

        if reponse_text is None:
            raise ValueError(f"Empty response from GPT. {messages=}, {response=}")

        return reponse_text, [*messages, {"role": "assistant", "content": reponse_text}]

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
                if re.findall(rf"\b{l}\b", line, flags=re.IGNORECASE)
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
            print(f"WARNING: Missing scores for some labels in {path}")
            print(f"Response: {response}")
            print()

        label_scores = {label: -0.1 for label in task_labels} | response_scores
        scores = [label_scores[label] for label in task_labels]
        if normalize:
            scores = t.Tensor(scores).softmax(0).tolist()
        label_probs = {label: prob for label, prob in zip(task_labels, scores)}

        # This writes directly into the cache object we've been passed
        cache[path] = label_probs

        return label_probs

    def _predict_video(self, item: dict, task: Task, cache: dict):
        path = item["path"]

        logging.info(f"Predicting task {task.id} for video: {path}")
        logging.info(f"True label: {item['labels'][task.id]}")
        if path in cache:
            logging.info(f"Using cached scores: {cache[path]}")
        else:
            logging.info("No cached scores found")
            system_prompt = [{"type": "text", "text": task.prompt_gpt}]
            history = [{"role": "system", "content": system_prompt}]

            heading = {"type": "text", "text": "Input frames:"}
            frames = [GPTModel._frame_to_payload(f) for f in item["frames"]]
            _, history = self._send_request([heading, *frames], history)

            class_list = "\n".join(
                [
                    f"- ({label}) {description}"
                    for label, description in list(task.label_prompts.items())
                ]
            )
            scoring_prompt = GPTModel.scoring_prompt.format(classes=class_list)
            answer, history = self._send_request(scoring_prompt, history)
            GPTModel.parse_and_cache_scores(answer, path, task.labels, cache)

        logging.info(f"Predicted scores: {cache[path]}")

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
                    "true_label_idx": task.labels.index(item["labels"][task.id]),
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
        scoring_prompt = GPTModel.scoring_prompt_v2.format(classes=class_list)

        heading = {"type": "text", "text": "Input frames:"}
        frames = [GPTModel._frame_to_payload(f) for f in item["frames"]]

        history = [
            {"role": "system", "content": task.prompt_gpt + scoring_prompt},
            {"role": "user", "content": [heading, *frames]},
        ]

        return history

    def predict(self, videos: list[Video], tasks: list[Task]) -> Optional[pl.DataFrame]:
        logging.info("Configuring dataset...")
        subsample = partial(utils.subsample, n_frames=self._n_frames)
        transforms = [subsample, utils.frames_to_b64]
        dataset_iter = VideoDataset(videos, tasks, transforms)
        # converting to a list forces all the videos to be converted up front, so that if any are invalid, an error will be thrown before any GPT-4 calls are made
        logging.info("Processing videos...")
        dataset = list(dataset_iter)
        logging.info("Predicting...")

        if self._async_batch:
            print("WARNING: Ignoring all cache when using async_batch")

            batch_ids = []

            for group in set("/".join(task.id.split("/")[:2]) for task in tasks):
                logging.info(f"Sending off batch file for group {group}")

                task_list = [
                    {
                        "custom_id": f"{item['path']},{task.id}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self._model,
                            "messages": self._build_batch_message(item, task),
                            "max_tokens": 2500,  # Hopefully not needed
                        },
                    }
                    for item in dataset
                    for task in tasks
                    if task.id.startswith(group) and item["labels"][task.id]
                ]

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

                logging.info(f"Created batch object: {batch_object} for group {group}")

                batch_ids.append(batch_object.id)

            logging.info(f"Created batches: {batch_ids}")

            return None

        print("WARNING: Evaluating synchronously")
        results = []
        for item in dataset:
            for task in [task for task in tasks if item["labels"][task.id]]:
                task_info = {
                    # "id": task.id,  # the task ID is not given to GPT-4, so it's safe to make the cache not depend on it
                    "gpt4_prompt": task.prompt_gpt,
                    "labels": [
                        description for description in task.label_prompts.values()
                    ],
                }
                model_info = {"id": self.id, "n_frames": self._n_frames}
                cache = utils.load_cache(self._cache_dir, task_info, model_info)
                results.append(self._predict_video(item, task, cache))
                utils.save_cache(cache, self._cache_dir, task_info, model_info)
        result = pl.concat(results)

        return result
