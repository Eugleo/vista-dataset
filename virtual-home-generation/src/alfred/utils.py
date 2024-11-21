from pathlib import Path
from typing import Callable, Hashable, TypeVar

from alfred.trajectory import Trajectory

T = TypeVar("T")


def group_by(objects: list[T], fun: Callable[[T], Hashable]) -> list[list[T]]:
    groups = {}
    for obj in objects:
        groups.setdefault(fun(obj), []).append(obj)
    result = list(groups.values())
    return sorted(result, key=lambda l: len(l), reverse=True)


def get_action_tuples(trajectory: Trajectory):
    return [(a.action, a.object1, a.object2) for a in trajectory.actions]


def set_action_tuples(trajectory: Trajectory, tuples):
    for a, t in zip(trajectory.actions, tuples):
        a.action, a.object1, a.object2 = t


def load_trajectories(metadata_path: Path) -> list[Trajectory]:
    trajectories = []
    for trajectory_path in metadata_path.glob("**/traj_data.json"):
        try:
            trajectory = Trajectory.from_file(trajectory_path)
        except Exception as e:
            print("Error loading", trajectory_path, e)
            continue
        trajectories.append(trajectory)
    return trajectories


def set_seed(seed: int) -> None:
    import os
    import random

    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
