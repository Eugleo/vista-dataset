from pathlib import Path
from typing import Annotated, Literal, TypedDict
import typer
import json
from rich.progress import track

from vlm.objects import Task


class DataEntry(TypedDict):
    path: str
    label: str


class Label(TypedDict):
    task: str
    label: str
    description: str


class FinalItem(TypedDict):
    video: str  # video path
    labels: list[Label]  # labels per task
    environment: Literal["virtual_home", "real_life", "minecraft"]
    n_steps: int
    group: str


def combine_items(items: list[FinalItem]) -> FinalItem:
    assert (
        len(set(i["video"] for i in items)) == 1
    ), "The items must all refer to the same video"

    return items[0] | {"labels": [l for i in items for l in i["labels"]]}


def entry_to_item(path: Path, task: Task, entry: DataEntry) -> FinalItem:
    label: list[Label] = {
        "task": task.id,
        "label": entry["label"],
        "description": task.label_prompts[entry["label"]],
    }

    if "alfred" in path.parts:
        _, environment, level_str, group, *_ = path.parts
        n_steps = 1 if level_str == "foundation" else int(level_str[-1])
    else:
        raise ValueError("Fill in your logic in entry_to_item")

    return {
        "video": entry["path"],
        "labels": [label],
        "environment": environment,
        "n_steps": n_steps,
        "group": group,
    }


def collapse(items: list[FinalItem]) -> list[FinalItem]:
    """
    Make sure there is at most one item per unique video by combining items with the same video.
    """
    items_by_video: dict[str, list[FinalItem]] = {}
    for item in items:
        items_by_video.setdefault(item["video"], []).append(item)

    result = [combine_items(items) for items in items_by_video.values()]

    return result


def main(
    task_dir: Annotated[
        Path,
        typer.Option(file_okay=False, dir_okay=True, readable=True),
    ],
    output: Annotated[
        Path, typer.Argument(file_okay=True, dir_okay=False, writable=True)
    ] = "dataset.json",
):
    items = []
    for data_file in track(
        list(task_dir.glob("*/**/*_data.json")), description="Loading data files..."
    ):
        with open(data_file) as f:
            task_id, task_path = (
                data_file.stem.replace("_data", ""),
                str(data_file).replace("_data.json", ".yaml"),
            )
            task = Task.from_file(task_id, task_path)
            for entry in json.load(f):
                items.append(entry_to_item(data_file, task, entry))

    with open(output, "w") as f:
        json.dump(items, f, indent=2)


if __name__ == "__main__":
    typer.run(main)
