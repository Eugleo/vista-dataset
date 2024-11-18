# This code converts the data from the format expected by our evaluation code
# to a more plain format that is easier to work with if you want to do your own analysis.

# %%
# %! load_ext autoreload
# %! autoreload 2

# %%

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from vlm.objects import Task


def _parse_level(yaml_path: Path) -> Optional[int]:
    """Parse level from yaml path. Returns 1 for foundation/extrapyramidal, or level_n number."""
    if (
        "foundation" in str(yaml_path).lower()
        or "extrapyramidal" in str(yaml_path).lower()
    ):
        return 1

    level = next(
        (int(p.split("_")[1]) for p in yaml_path.parts if p.startswith("level_")), None
    )
    if level is None:
        logging.warning(f"Could not parse level from path: {yaml_path}")
    return level


def _parse_environment(yaml_path: Path) -> Optional[str]:
    """Parse environment from yaml path (alfred/real_life/minecraft)."""
    if "alfred" in str(yaml_path).lower():
        return "alfred"
    elif "real_life" in str(yaml_path).lower():
        return "real_life"
    elif "minecraft" in str(yaml_path).lower():
        return "minecraft"
    else:
        logging.warning(f"Unknown environment in path: {yaml_path}")
        return None


def _process_single_label(
    label: str,
    task: Task,
    video_path: str,
    level: int,
    environment: str,
    problem_set_type: str,
) -> Optional[dict]:
    """Process a single label and return video entry dict if valid."""
    if label not in task.label_prompts:
        logging.warning(f"Label {label} not found in task {task.id}")
        return None

    return {
        "video": str(Path(environment) / Path(video_path)),
        "level": level,
        "description": task.label_prompts[label],
        "environment": environment,
        "problem_set_type": problem_set_type,
        "problem_set": task.id,
    }


def _process_video_entry(
    entry: Dict[str, Any],
    task: Task,
    yaml_path: Path,
) -> List[Dict[str, Any]]:
    """Process a single video entry, returning list of video data dicts."""
    video_entries = []

    # Parse required metadata
    level = _parse_level(yaml_path)
    environment = _parse_environment(yaml_path)
    if level is None or environment is None:
        return []

    problem_set_type = yaml_path.parts[-2]
    video_path = entry["path"]
    label = entry["label"]

    # Handle comma-separated labels
    labels = label.split(",") if "," in label else [label]
    for label in labels:
        entry_dict = _process_single_label(
            label.strip(), task, video_path, level, environment, problem_set_type
        )
        if entry_dict:
            video_entries.append(entry_dict)

    return video_entries


def load_video_dataset(tasks_dir: str = "../tasks/") -> pl.DataFrame:
    """
    Load all tasks and their associated videos from yaml/json pairs in the tasks directory.

    Args:
        tasks_dir: Root directory containing task yaml files and associated json data files

    Returns:
        Polars DataFrame containing video information with columns:
        - video (path)
        - level (parsed from path)
        - description (task description for the video)
        - environment (alfred/real_life/minecraft)
        - problem_set_type (from path)
        - problem_set (task id)
    """
    tasks_root = Path(tasks_dir)
    if not tasks_root.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_root}")

    video_data: List[Dict[str, Any]] = []

    # Find all yaml files recursively
    for yaml_path in tasks_root.rglob("*.yaml"):
        try:
            # Get corresponding json path
            json_path = yaml_path.with_stem(f"{yaml_path.stem}_data").with_suffix(
                ".json"
            )
            if not json_path.exists():
                logging.warning(f"Missing json file for task: {yaml_path}")
                continue

            # Load task and video data
            task = Task.from_file(str(yaml_path), yaml_path)
            with open(json_path) as f:
                video_entries = json.load(f)

            # Process each video entry
            for entry in video_entries:
                video_data.extend(_process_video_entry(entry, task, yaml_path))

        except Exception as e:
            logging.error(f"Error processing {yaml_path}: {str(e)}")
            continue

    if not video_data:
        raise ValueError("No valid video data found in tasks directory")

    return pl.DataFrame(video_data)


# %%
dataset = load_video_dataset()
dataset.write_csv("../metadata.csv")

# %%
