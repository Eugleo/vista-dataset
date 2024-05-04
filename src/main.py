from pathlib import Path
from typing import Annotated, Optional

import polars as pl
import typer
from polars import col as c
from rich.progress import Progress, SpinnerColumn, TextColumn
from vlm import plots, utils
from vlm.config import ExperimentConfig

app = typer.Typer()


@app.command()
def evaluate(config: Annotated[str, typer.Argument()]):
    experiment_config = ExperimentConfig.from_file(config)
    experiment = ExperimentConfig.to_experiment(experiment_config)
    experiment.run()


@app.command()
def plot_habitat(
    experiment: Annotated[Optional[str], typer.Argument()] = None,
    interactive: Annotated[bool, typer.Option()] = False,
    experiment_dir: Annotated[
        str, typer.Option()
    ] = "/data/datasets/vlm_benchmark/experiments",
):
    if experiment is None:
        experiments = Path(experiment_dir).iterdir()
        dir = sorted(experiments, key=lambda d: d.stat().st_mtime)[-1]
    else:
        dir = Path(experiment_dir) / experiment

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        progress.add_task("Creating plots...")
        df = pl.read_json(dir / "results.json")

        predictions = utils.get_predictions(df, standardize=True)
        metrics = predictions.group_by(["task", "model"]).agg(
            accuracy=utils.compute_metric(utils.accuracy)
        )

        groups = {
            "Proximity": ["walk_to_chair", "walk_to_plant", "walk_to_tv"],
            "Rooms": ["recognize_room_model", "recognize_room_scan"],
            "Heading into a Specific Room": ["find_room"],
            "Objects": [
                "recognize_small_object",
                "recognize_large_object",
            ],
            "Large Objects in a Room": ["recognize_large_object_in_context"],
            "Containers": [
                "recognize_apple_container",
                "recognize_can_container",
                "recognize_hammer_container",
            ],
            "Small Objects in a Container": ["recognize_small_object_in_container"],
            "Container State": ["recognize_container"],
            "Opening Motion": ["open_cabinet", "open_fridge", "open_drawer"],
            "Walking to a concrete Object": ["walk_to"],
            "Moving an object From and To a Specific Container": ["move_can"],
        }
        for name, tasks in groups.items():
            baselines = {
                task: 1
                / df.filter(c("task") == task).get_column("true_label").n_unique()
                for task in tasks
            }
            plot = plots.task_performance(
                metrics.filter(pl.col("task").is_in(tasks)),
                predictions.filter(pl.col("task").is_in(tasks)),
                metric="accuracy",
                title=name,
                baselines=baselines,
            )
            plot_dir = dir / "plots"
            plot_dir.mkdir(exist_ok=True, parents=True)
            plot.write_image(plot_dir / f"{name}_performance.pdf")
        print("Plots saved")
