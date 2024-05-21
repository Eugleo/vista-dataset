import json
import logging
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Annotated, Optional

import dotenv
import jsonlines
import openai
import polars as pl
import typer
from polars import col as c
from rich.pretty import pprint
from rich.progress import Progress, SpinnerColumn, TextColumn
from vlm import plots, utils
from vlm.config import ExperimentConfig
from vlm.models import GPTModel
from vlm.objects import Task

app = typer.Typer()


@app.command()
def cancel_batch(
    batch_ids: Annotated[list[str], typer.Argument()],
):
    dotenv.load_dotenv()
    client = openai.Client()
    for batch_id in batch_ids:
        client.batches.cancel(batch_id)
        print(f"Batch {batch_id} is cancelled.")


@app.command()
def download_batch(
    batch_ids: Annotated[list[str], typer.Argument()],
    experiment: Annotated[Optional[str], typer.Option()] = None,
    experiment_dir: Annotated[
        str, typer.Option()
    ] = "/data/datasets/vlm_benchmark/experiments",
    task_dir: Annotated[str, typer.Option()] = "/data/datasets/vlm_benchmark/tasks",
    video_dir: Annotated[str, typer.Option()] = "/data/datasets/vlm_benchmark/videos",
    verbose: Annotated[bool, typer.Option()] = False,
):
    dotenv.load_dotenv()
    client = openai.Client()
    if experiment is None:
        experiments = [d for d in Path(experiment_dir).iterdir() if d.is_dir()]
        dir = sorted(experiments, key=lambda d: d.stat().st_mtime)[-1]
        print(f"Loading the most recent experiment: {dir}...")
    else:
        dir = Path(experiment_dir) / experiment

    task_true_labels = {}

    for batch_id in batch_ids:
        batch_file = client.batches.retrieve(batch_id)

        if batch_file.status != "completed":
            print(f"{batch_id}: {batch_file.status}")
            if verbose:
                pprint(batch_file)
            continue

        assert batch_file.output_file_id is not None
        batch_response_dir = dir / "batch_responses"
        batch_response_dir.mkdir(exist_ok=True, parents=True)
        batch_reponse_file = batch_response_dir / f"{batch_id}.jsonl"
        if not batch_reponse_file.exists():
            print(f"Downloading batch response for {batch_id}...")
            content = client.files.content(batch_file.output_file_id)
            content.write_to_file(batch_reponse_file)
        else:
            print(f"{batch_id}: downloaded")

        results = []
        with jsonlines.open(batch_reponse_file) as reader:
            for response in reader:
                model = response["response"]["body"]["model"]
                if response["response"]["status_code"] != 200:
                    print(f"WARNING: Request failed: {response}")
                    continue
                path, task_id = response["custom_id"].split(",")
                answer = response["response"]["body"]["choices"][0]["message"][
                    "content"
                ]

                if task_id not in task_true_labels:
                    task = Task.from_file(task_id, Path(task_dir) / f"{task_id}.yaml")
                    with open(Path(task_dir) / f"{task_id}_data.json") as f:
                        true_labels = {
                            str(Path(video_dir) / obj["path"]): obj["label"]
                            for obj in json.load(f)
                        }
                        task_true_labels[task_id] = true_labels
                true_labels = task_true_labels[task_id]

                label_scores = GPTModel.parse_and_cache_scores(
                    answer, path, task.labels, cache={}, verbose=verbose
                )

                results += [
                    {
                        "task": task_id,
                        "model": model,
                        "video": path,
                        "label": label,
                        "label_idx": label_idx,
                        "score": score,
                        "true_label": true_labels[path],
                        "true_label_idx": task.labels.index(true_labels[path]),
                    }
                    for label_idx, (label, score) in enumerate(label_scores.items())
                ]

        results = pl.DataFrame(results)
        result_dir = dir / "results"
        result_dir.mkdir(exist_ok=True, parents=True)
        results.write_json(result_dir / f"{model}_{batch_id}.json")


@app.command()
def evaluate(config: Annotated[str, typer.Argument()]):
    experiment_config = ExperimentConfig.from_file(config)
    experiment = ExperimentConfig.to_experiment(experiment_config)
    experiment_dir = Path(experiment.output_dir) / experiment.id
    experiment_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        filename=experiment_dir / "log.txt",
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    experiment.run()


@app.command()
def plot_alfred(
    experiment: Annotated[Optional[str], typer.Argument()] = None,
    experiment_dir: Annotated[
        str, typer.Option()
    ] = "/data/datasets/vlm_benchmark/experiments",
    task_dir: Annotated[str, typer.Option()] = "/data/datasets/vlm_benchmark/tasks",
):
    if experiment is None:
        experiments = [d for d in Path(experiment_dir).iterdir() if d.is_dir()]
        dir = sorted(experiments, key=lambda d: d.stat().st_mtime)[-1]
        print(f"Loading the most recent experiment: {dir}...")
    else:
        dir = Path(experiment_dir) / experiment

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        progress.add_task("Creating plots...")
        df = pl.concat(
            [pl.read_json(file) for file in (dir / "results").glob("*.json")]
        )

        scores = utils.standardize(df)

        task_labels = {}
        per_label_baselines = []
        for t in df.get_column("task").unique():
            task_labels[t] = Task.from_file(t, Path(task_dir) / f"{t}.yaml").labels
            with open(Path(task_dir) / f"{t}_data.json") as f:
                label_counts = Counter(v["label"] for v in json.load(f))
            for l in task_labels[t]:
                per_label_baselines.append(
                    {
                        "task": t,
                        "model": "Ω random",
                        "AP": label_counts[l] / sum(label_counts.values()),
                    }
                )
        per_label_baselines = pl.DataFrame(per_label_baselines)

        per_label_metrics = (
            # This sort is very important
            scores.sort("task", "model", "video", "label")
            .group_by("task", "model")
            .agg(
                AP=pl.struct("task", "label", "score", "true_label").map_batches(
                    partial(plots.average_precision, task_labels)
                )
            )
            .explode("AP")
            .sort("task", "model")
        )
        per_label_metrics = pl.concat([per_label_metrics, per_label_baselines])
        per_label_metrics.write_csv(dir / "per_label_metrics.csv")

        predictions = utils.get_predictions(scores)
        problems = plots.incorrect_video_labels(predictions)
        plot_dir = dir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        problems.write_csv(plot_dir / "problems.csv")

        def get_tasks(prefix):
            return [t for t in df.get_column("task").unique() if t.startswith(prefix)]

        groups = {
            "The whole Foundation Level": get_tasks("foundation"),
            "Object Recognition": get_tasks("foundation/objects"),
            "Container Recognition": get_tasks("foundation/containers"),
            "State: On v. Off": get_tasks("foundation/on_v_off"),
            "State: Sliced v. Whole": get_tasks("foundation/sliced_v_whole"),
            "Action: Cleaning": get_tasks("foundation/clean"),
            "Action: Heating": get_tasks("foundation/heat"),
            "Action: Cooling": get_tasks("foundation/cool"),
            "Action: Putting down v. Picking up": get_tasks("foundation/pick_v_put"),
            "Action: Slicing": get_tasks("foundation/slice"),
            "Action: Toggling On v. Off": get_tasks("foundation/toggle"),
            "Level 3": get_tasks("level_3"),
            "Level 6": get_tasks("level_6"),
        }

        for name, tasks in groups.items():
            filename = name.replace(": ", "_").replace(" ", "-")
            plot = plots.map_plot(
                per_label_metrics.filter(pl.col("task").is_in(tasks)),
                f"{name} (standardized)",
            )
            plot_dir = dir / "plots"
            plot_dir.mkdir(exist_ok=True, parents=True)
            plot.write_image(plot_dir / f"{filename}_mAP.pdf")

            plot = plots.overall_performance(
                per_label_metrics.filter(pl.col("task").is_in(tasks)),
                metric="AP",
                metric_label="Mean AP over all labels",
                title=f"{name} (standardized)",
            )
            plot.write_image(plot_dir / f"{filename}_mAP, overall.pdf")

            if "Slicing" in name:
                plot = plots.task_performance(
                    per_label_metrics.filter(pl.col("task").is_in(tasks))
                    .group_by("task", "model")
                    .agg(mAP=c("AP").mean()),
                    predictions.filter(pl.col("task").is_in(tasks)),
                    metric="mAP",
                    title=f"{name} (standardized)",
                    baselines=None,
                    labels=task_labels,
                    tasks=tasks,
                )
                plot.write_image(plot_dir / f"{filename}_details.pdf")


@app.command()
def plot(
    experiment: Annotated[Optional[str], typer.Argument()] = None,
    experiment_dir: Annotated[
        str, typer.Option()
    ] = "/data/datasets/vlm_benchmark/experiments",
    task_dir: Annotated[str, typer.Option()] = "/data/datasets/vlm_benchmark/tasks",
):
    if experiment is None:
        experiments = [d for d in Path(experiment_dir).iterdir() if d.is_dir()]
        dir = sorted(experiments, key=lambda d: d.stat().st_mtime)[-1]
        print(f"Loading the most recent experiment: {dir}...")
    else:
        dir = Path(experiment_dir) / experiment

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        progress.add_task("Creating plots...")
        df = pl.read_json(dir / "results.json")

        scores = utils.standardize(df)
        predictions = utils.get_predictions(scores)
        metrics = (
            predictions.group_by(["task", "model"])
            .agg(accuracy=utils.compute_metric(utils.accuracy))
            .sort("task", "model")
        )

        problems = plots.incorrect_video_labels(predictions)
        plot_dir = dir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        problems.write_csv(plot_dir / "problems.csv")

        group_names = set(t.split("/")[1] for t in df.get_column("task").unique())
        groups = {
            name: [t for t in df.get_column("task").unique() if t.startswith(name)]
            for name in group_names
        }

        for name, tasks in groups.items():
            tasks = [t for t in tasks if len(df.filter(c("task") == t)) > 0]
            if not tasks:
                continue
            labels = {
                t: list(
                    Task.from_file(t, Path(task_dir) / f"{t}.yaml").label_prompts.keys()
                )
                for t in tasks
            }
            baselines = {t: 1 / len(labels[t]) for t in tasks for t in tasks}
            plot = plots.task_performance(
                metrics.filter(pl.col("task").is_in(tasks)),
                predictions.filter(pl.col("task").is_in(tasks)),
                metric="accuracy",
                title=f"{name}",
                baselines=baselines,
                labels=labels,
                tasks=tasks,
            )
            plot_dir = dir / "plots"
            plot_dir.mkdir(exist_ok=True, parents=True)
            plot.write_image(plot_dir / f"{name}_performance.pdf")

            plot = plots.overall_performance(
                metrics.filter(pl.col("task").is_in(tasks)),
                metric="accuracy",
                title=f"{name} (standardized)",
            )
            plot.write_image(plot_dir / f"{name}_overall.pdf")

        print("Plots seed")
