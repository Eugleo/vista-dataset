import os
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd

def parse_args():
    parser = ArgumentParser(prog="compose_dataset", description="Compose per-video .json labeling into one dataframe.")
    parser.add_argument("-s", "--source-dirs", nargs="+", help="Directories with .json files", required=True)

    return parser.parse_args()

def load_json(filename) -> Dict[str, Any]:
    with open(filename, "r") as f:
        return json.load(f)

def binarize_dict(data: Dict[str, Any]) -> Dict[str, bool]:
    #return {k: "yes" if bool(v) else "no" for k, v in data.items()}
    return {k: bool(v) for k, v in data.items()}

def count_labels(label: str) -> int:
    if len(label) == 0:
        return 0
    return len(label.split(","))

def process_dict(data: Dict[str, Any]) -> Dict[str, int]:
    return {k: count_labels(v) for k, v in data.items()}

def collect_dir(directory: str, video_extention: str = ".mp4") -> pd.DataFrame:
    source_dir = Path(directory)
    freeform = []
    tasks = []
    names = []

    for filename in source_dir.iterdir():
        if not filename.suffix == ".json":
            continue
        
        data = load_json(filename)
        tasks.append(binarize_dict(data["tasks"]))
        freeform.append(data["freeform_description"])
        names.append(str(filename.with_suffix(video_extention)))

    data = pd.DataFrame(tasks)
    data["freeform_description"] = freeform
    data["video_path"] = names

    return data

def make_dataframe(source_dirs):
    data = pd.DataFrame()
    for src in source_dirs:
        data = pd.concat([data, collect_dir(src)])

    print(len(data), "samples")
    print("Positives per task:")
    print(data.drop(columns=["freeform_description", "video_path"]).sum())

    # print(data)
    #print("="*70)
    #print(freeform)

    return data

def create_task_data_file(task: str, data: pd.DataFrame):
    task_dir = Path("tasks/minecraft")
    task_dir.mkdir(parents=True, exist_ok=True)
    output_path = task_dir / f"{task}_data.json"
    with open(output_path, "w") as f:
        pairs = data[["video_path", task]]\
            .rename(columns={"video_path": "path", task: "label"})\
            .to_dict("records")
        json.dump(pairs, f, indent=2)
    return

def main():
    """
    Input: a list of directories with videos and paired json files with labels.
    Output: a file structure suitable for vlm-benchmark codebase. Concretely:
    - tasks/minecraft/{task_name}.yaml          files, defining the task (gpt prompt, labels with corresponding prompts, etc)
    - tasks/minecraft/{task_name}_data.json     a list of dicts of form {"path": "path/to/video", "label": "this_video_label"}
    """
    args = parse_args()
    data = make_dataframe(args.source_dirs)

    tasks = data.drop(columns=["freeform_description", "video_path"]).columns

    # for task in tasks:
    #     create_task_data_file(task, data)
    
    true_label = data.drop(columns=["freeform_description", "video_path"])\
        .apply(lambda row: ",".join(task for task, val in zip(tasks, row) if val), 1)

    data["true_text_label"] = true_label

    print(data)

    task_dir = Path("tasks/minecraft")
    task_dir.mkdir(parents=True, exist_ok=True)
    output_path = task_dir / f"multilabel_data.json"
    with open(output_path, "w") as f:
        pairs = data[["video_path", "true_text_label"]]\
            .rename(columns={"video_path": "path", "true_text_label": "label"})\
            .to_dict("records")
        json.dump(pairs, f, indent=2)


if __name__ == "__main__":
    # make_dataframe(parse_args().source_dirs)
    main()