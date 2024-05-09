import logging
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
def plot_habitat(
    experiment: Annotated[Optional[str], typer.Argument()] = None,
    interactive: Annotated[bool, typer.Option()] = False,
    experiment_dir: Annotated[
        str, typer.Option()
    ] = "/data/datasets/vlm_benchmark/experiments",
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

        predictions = utils.get_predictions(df, standardize=True)
        metrics = predictions.group_by(["task", "model"]).agg(
            accuracy=utils.compute_metric(utils.accuracy)
        )

        problems = plots.incorrect_video_labels(predictions)
        plot_dir = dir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        problems.write_csv(plot_dir / "problems.csv")

        groups = {
            "Closed v. Closing": [
                "closed_or_closing/cabinet",
                "closed_or_closing/drawer",
                "closed_or_closing/fridge",
            ],
            "Open v. Opening": [
                "open_or_opening/cabinet",
                "open_or_opening/drawer",
                "open_or_opening/fridge",
            ],
            "Open v. Closed": [
                "open_or_closed/cabinet",
                "open_or_closed/drawer",
                "open_or_closed/fridge",
            ],
            "Opening v. Closing": [
                "opening_or_closing/cabinet",
                "opening_or_closing/drawer",
                "opening_or_closing/fridge",
            ],
            "Opening v. Closing Specific Container": ["opening_or_closing/container"],
            "Container Type": [
                "container_type/apple",
                "container_type/can",
                "container_type/hammer",
            ],
            "Near v. Far": ["near_or_far/chair", "near_or_far/plant", "near_or_far/tv"],
            "Object": ["object/large", "object/small"],
            "Object in Context": ["object/large_in_room", "object/small_in_container"],
            "Room": ["room/model", "room/scan"],
            "Find Room": ["room/find"],
            "Sequence of Rooms": ["room/sequence"],
            "Move Can": ["move_can"],
        }

        for name, tasks in groups.items():
            tasks = [t for t in tasks if len(df.filter(c("task") == t)) > 0]
            if not tasks:
                continue
            baselines = {
                task: 1
                / df.filter(c("task") == task).get_column("true_label").n_unique()
                for task in tasks
            }
            plot = plots.task_performance(
                metrics.filter(pl.col("task").is_in(tasks)),
                predictions.filter(pl.col("task").is_in(tasks)),
                metric="accuracy",
                title=f"{name} (standardized)",
                baselines=baselines,
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
