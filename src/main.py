from pathlib import Path
from typing import Annotated, Optional

import polars as pl
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from vlm import utils
from vlm.config import ExperimentConfig

app = typer.Typer()


@app.command()
def evaluate(config: Annotated[str, typer.Argument()]):
    experiment_config = ExperimentConfig.from_file(config)
    experiment = ExperimentConfig.to_experiment(experiment_config)
    experiment.run()


@app.command()
def plot(
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
        df = pl.read_csv(dir / "results.csv")

        predictions = utils.get_predictions(df)
        metrics = predictions.group_by(["task", "model"]).agg(
            mcc=utils.compute_metric(utils.mcc)
        )

        plot_dir = dir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)

        performance_plot = utils.performance_per_task(metrics)
        performance_plot.write_image(plot_dir / "task_performance.pdf")
        print("Performance plot saved")
        if interactive:
            performance_plot.show()

        for task in df["task"].unique().to_list():
            for model in df["model"].unique().to_list():
                task_df = predictions.filter(pl.col("task") == task).filter(
                    pl.col("model") == model
                )
                confusion_matrix = utils.confusion_matrix(task_df)
                confusion_matrix.figure_.set_figwidth(12)
                confusion_matrix.figure_.set_figheight(12)
                confusion_matrix.figure_.savefig(
                    plot_dir / f"{task}_{model}_confusion_matrix.pdf"
                )
        print("Confusion matrices saved")
