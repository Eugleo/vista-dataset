import json
import logging
from functools import partial
from pathlib import Path
from typing import Annotated, Optional

import dotenv
import jsonlines
import openai
import polars as pl
import typer
from polars import col as c
from rich.progress import Progress, SpinnerColumn, TextColumn
from vlm import plots, utils
from vlm.config import ExperimentConfig
from vlm.models import GPTModel
from vlm.objects import Task

app = typer.Typer()


@app.command()
def download_batch(
    batch_id: Annotated[str, typer.Argument()],
    experiment: Annotated[Optional[str], typer.Option()] = None,
    experiment_dir: Annotated[
        str, typer.Option()
    ] = "/data/datasets/vlm_benchmark/experiments",
    task_dir: Annotated[str, typer.Option()] = "/data/datasets/vlm_benchmark/tasks",
    video_dir: Annotated[str, typer.Option()] = "/data/datasets/vlm_benchmark/videos",
):
    dotenv.load_dotenv()
    client = openai.Client()
    batch_file = client.batches.retrieve(batch_id)

    if batch_file.status != "completed":
        print(f"Batch {batch_id} is not completed yet. Status: {batch_file.status}")
        return

    if experiment is None:
        experiments = [d for d in Path(experiment_dir).iterdir() if d.is_dir()]
        dir = sorted(experiments, key=lambda d: d.stat().st_mtime)[-1]
        print(f"Loading the most recent experiment: {dir}...")
    else:
        dir = Path(experiment_dir) / experiment

    assert batch_file.output_file_id is not None
    content = client.files.content(batch_file.output_file_id)
    content.write_to_file(dir / "batch_response.jsonl")

    results = []
    with jsonlines.open(dir / "batch_response.jsonl") as reader:
        for response in reader:
            model = response["response"]["body"]["model"]
            if response["response"]["status_code"] != 200:
                print(f"WARNING: Request failed: {response}")
                continue
            path, task_id = response["custom_id"].split(",")
            answer = response["response"]["body"]["choices"][0]["message"]["content"]

            task = Task.from_file(task_id, Path(task_dir) / f"{task_id}.yaml")
            with open(Path(task_dir) / f"{task_id}_data.json") as f:
                true_labels = {
                    str(Path(video_dir) / obj["path"]): obj["label"]
                    for obj in json.load(f)
                }

            label_scores = GPTModel.parse_and_cache_scores(
                answer, path, task.labels, cache={}
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
    results.write_json(result_dir / f"{model}.json")


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
        scores = utils.add_random_baseline(scores)

        task_labels = {
            t: list(
                Task.from_file(t, Path(task_dir) / f"{t}.yaml").label_prompts.keys()
            )
            for t in df.get_column("task").unique()
        }

        per_label_metrics = (
            # This sort is very important
            scores.sort("task", "model", "video", "label")
            .group_by("task", "model")
            .agg(
                mAP=pl.struct("task", "label", "score", "true_label").map_batches(
                    partial(plots.mean_average_precision, task_labels)
                )
            )
            .explode("mAP")
            .sort("task", "model")
        )
        per_label_metrics.write_csv(dir / "per_label_metrics.csv")

        predictions = utils.get_predictions(df)
        problems = plots.incorrect_video_labels(predictions)
        plot_dir = dir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        problems.write_csv(plot_dir / "problems.csv")

        clean_tasks = [
            t
            for t in df.get_column("task").unique()
            if t.startswith("foundation/clean")
        ]
        heat_tasks = [
            t for t in df.get_column("task").unique() if t.startswith("foundation/heat")
        ]
        cool_tasks = [
            t for t in df.get_column("task").unique() if t.startswith("foundation/cool")
        ]

        groups = {
            "The whole Foundation Level": [
                "foundation/objects/pick_from_counter_top",
                "foundation/objects/pick_from_dining_table",
                "foundation/objects/pick_from_somewhere",
                "foundation/containers/place_butter_knife",
                "foundation/containers/place_soap_bar",
                "foundation/containers/place_mug",
                "foundation/containers/place_key_chain",
            ]
            + clean_tasks
            + heat_tasks
            + cool_tasks,
            "Cleaning v. just putting into a sink": clean_tasks,
            "Heating v. just putting into a microwave": heat_tasks,
            "Cooling v. just standing in front of a fridge": cool_tasks,
            "Object Recognition": [
                "foundation/objects/pick_from_counter_top",
                "foundation/objects/pick_from_dining_table",
                "foundation/objects/pick_from_somewhere",
            ],
            "Container Recognition": [
                "foundation/containers/place_butter_knife",
                "foundation/containers/place_soap_bar",
                "foundation/containers/place_mug",
                "foundation/containers/place_key_chain",
            ],
            # "Location Recognition, kinda broken": [
            #     "level_1/object_recognition/goto/location"
            # ],
        }

        for name, tasks in groups.items():
            plot = plots.map_plot(
                per_label_metrics.filter(pl.col("task").is_in(tasks)),
                f"{name} (standardized)",
            )
            plot_dir = dir / "plots"
            plot_dir.mkdir(exist_ok=True, parents=True)
            plot.write_image(plot_dir / f"{name} mAP.pdf")

            plot = plots.overall_performance(
                per_label_metrics.filter(pl.col("task").is_in(tasks)),
                metric="mAP",
                metric_label="Mean AP over all labels",
                title=f"{name} (standardized)",
            )
            plot.write_image(plot_dir / f"{name} mAP, overall.pdf")


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
