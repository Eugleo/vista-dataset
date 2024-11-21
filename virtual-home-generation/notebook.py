# %%
#!%load_ext autoreload
#!%autoreload 2

# %%
import math
import random
from pathlib import Path

from plotnine import *
from rich import print as rprint

from alfred import utils
from alfred.task import (
    clean_tasks,
    container_tasks,
    cool_tasks,
    heat_tasks,
    object_tasks,
    on_v_off_tasks,
    permutation_task,
    pick_up_tasks,
    remixed_task,
    slice_tasks,
    sliced_v_whole_tasks,
    toggle_tasks,
    write_config,
)
from alfred.trajectory import shorten_trajectories

utils.set_seed(42)

# %%
trajectories = utils.load_trajectories(Path("metadata"))
groups_per_level = {
    level: sorted(
        utils.group_by(
            shorten_trajectories(trajectories, level), lambda t: tuple(t.actions)
        ),
        key=lambda g: len(g),
        reverse=True,
    )
    for level in range(11)
}

# %%
benchmark_path = Path("test_2")
easy_tasks = [t for t in trajectories if t.type != "pick_and_place_with_movable_recep"]

# %%
tasks = []


def limit(tasks, count=5):
    random.shuffle(tasks)
    return tasks[:count]


for t in limit(pick_up_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

for t in limit(clean_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

for t in object_tasks(easy_tasks):
    tasks.append(t.write(benchmark_path))

for t in container_tasks(easy_tasks):
    tasks.append(t.write(benchmark_path))

for t in limit(heat_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

for t in limit(cool_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

for t in limit(toggle_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

for t in limit(slice_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

for t in limit(sliced_v_whole_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

for t in limit(on_v_off_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))


toggle_scenes = set(
    t.actions[0].scene
    for t in trajectories
    if any(a.action == "ToggleObject" for a in t.actions)
)

permutation_tasks_per_level = 33
num_classes = 3
for level in range(2, 9):
    level_num_classes = min(num_classes, math.factorial(level))
    groups = groups_per_level[level]
    random.shuffle(groups)

    groups = groups_per_level[level]
    normal_groups, toggle_groups = [], []
    for g in groups:
        if any(a.scene in toggle_scenes for a in g[0].actions):
            toggle_groups.append(g)
        else:
            normal_groups.append(g)
    random.shuffle(toggle_groups)
    random.shuffle(normal_groups)

    candidates = {}

    consistent_candidates, inconsistent_candidates = [], []
    for g in toggle_groups:
        seed = [t for t in g if t.actions[0].scene in toggle_scenes][0]
        try:
            task = permutation_task(g[0], level_num_classes, trajectories)
        except:
            continue
        if len(task.trajectories) == level_num_classes:
            assert task.metadata is not None
            if task.metadata["is_consistent_wrt_slicing"]:
                consistent_candidates.append(task)
            else:
                inconsistent_candidates.append(task)
        if len(consistent_candidates) >= 8:
            break

    g = sorted(
        groups,
        key=lambda g: not any(
            t.action in ["HeatObject", "CoolObject", "CoolObject"] for t in g[0].actions
        ),
    )

    for g in groups:
        seed = random.choice(g)
        num_results = len(consistent_candidates + inconsistent_candidates)

        try:
            task = permutation_task(g[0], level_num_classes, trajectories)
        except:
            continue

        if len(task.trajectories) == level_num_classes:
            assert task.metadata is not None
            if task.metadata["is_consistent_wrt_slicing"]:
                consistent_candidates.append(task)
            else:
                inconsistent_candidates.append(task)
        if len(consistent_candidates) >= permutation_tasks_per_level:
            break
    else:
        print(
            f"Not enough consistent tasks for level {level}: {len(consistent_candidates)}"
        )
        to_add = permutation_tasks_per_level - len(consistent_candidates)
        print(f"Adding {to_add} inconsistent tasks")
        if to_add > len(inconsistent_candidates):
            print(
                f"Only writing {len(consistent_candidates) + len(inconsistent_candidates)} out of {permutation_tasks_per_level} tasks"
            )

    for task in (consistent_candidates + inconsistent_candidates)[
        :permutation_tasks_per_level
    ]:
        name = task.write(benchmark_path)
        # rprint([t.description for t in task.trajectories])
        tasks.append(name)
        print(name)


videos_per_task = 8

level_to_num_videos = {
    2: (4, 4),
    3: (2, 3),
    4: (2, 2),
    5: (4, 1),
    6: (3, 1),
    7: (2, 1),
    8: (1, 1),
}

assert all(
    ((a + (level - 1) * b) == videos_per_task)
    for level, (a, b) in level_to_num_videos.items()
)

remix_tasks_per_level = 12

for level in range(2, 9):
    videos_for_prefix_0, videos_per_prefix = level_to_num_videos[level]
    groups = groups_per_level[level]
    normal_groups, toggle_groups = [], []
    for g in groups:
        if any(a.scene in toggle_scenes for a in g[0].actions):
            toggle_groups.append(g)
        else:
            normal_groups.append(g)
    random.shuffle(toggle_groups)
    random.shuffle(normal_groups)

    candidates = {}

    for g in toggle_groups:
        seed = [t for t in g if t.actions[0].scene in toggle_scenes][0]
        task = remixed_task(seed, videos_for_prefix_0, videos_per_prefix, trajectories)
        candidates.setdefault(len(task.trajectories), []).append(task)
        if len(candidates.get(videos_per_task + 1, [])) >= 3:
            break

    for g in toggle_groups + normal_groups:
        g = sorted(g, key=lambda t: t.actions[0].scene not in toggle_scenes)
        seed = g[0]
        num_results = len(candidates.get(videos_per_task + 1, []))

        if num_results <= 6:
            if not any(a.action == "HeatObject" for a in seed.actions):
                continue
        elif num_results <= 9:
            if not any(a.action == "CoolObject" for a in seed.actions):
                continue
        elif num_results <= 12:
            if not any(a.action == "CleanObject" for a in seed.actions):
                continue

        task = remixed_task(seed, videos_for_prefix_0, videos_per_prefix, trajectories)

        # +1 because we have 8 remixed classes on top of the 1 original class
        candidates.setdefault(len(task.trajectories), []).append(task)
        if len(candidates.get(videos_per_task + 1, [])) >= remix_tasks_per_level:
            break
    else:
        print(
            f"Not enough long tasks for level {level}: {len(candidates.get(videos_per_task + 1, []))}"
        )
        lens = {
            i: len(candidates.get(i, []))
            for i in list(range(videos_per_task + 1, 1, -1))
        }
        print(f"Adding shorter tasks: {lens}")

    final_candidates = sum(
        [candidates.get(i, []) for i in list(range(videos_per_task + 1, 1, -1))], []
    )

    for task in final_candidates[:remix_tasks_per_level]:
        name = task.write(benchmark_path)
        # rprint([t.description for t in task.trajectories])
        tasks.append(name)
        print(name)


write_config(tasks, benchmark_path)

# %%
