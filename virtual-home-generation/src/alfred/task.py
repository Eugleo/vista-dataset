import itertools
import json
import random
import uuid
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Generator, Optional

import yaml
from rich.progress import track

from alfred import utils
from alfred.object import plausible_action_objects
from alfred.trajectory import HighLevelAction, Trajectory

HIGH_LEVEL_TASK_PROMPT = """
Your task is to describe what you see in each frame, separately, in a list. For each frame, concisely describe the scene, the objects we're holding, and from that deduce what actions have likely been performed since the last frame.
""".strip()


HIGH_LEVEL_TASK_EXAMPLE = """
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the very bottom of the screen and close to the camera, without any visible hands. Similarly, no hands are shown for cleaning, heating, or any other action.
- No action happens before the first frame. If the first frame starts with us already holding an object, it was already in our hands from the beginning.
- The objects almost never lie on the floor. If the object is at the bottom of the screen, close to the camera and looks like it is lying on the floor, we are most likely just holding it.
- The possible actions that can be performed are: pickup(object, location), put(object, location), heat(object, microwave), cool(object, fridge), clean(object, sink), slice(object), toggle(object), and goto(location). You should formulate your descriptions of what happened since the last frame based on these actions.
- To be able to `put` something, we first need to `pick` it up (or start with it in our hands).
- If we had something in hand in a frame, and it is not visible in the next frame, we must have performed a `put` action.
- If we didn't have anything in hand in a frame, and something appears in our hands in the next frame, we must have performed a `pick` action
- Slicing can only happen if we have a knife in our hands
- Cleaning involves putting an object in a sink, running water over it, and then picking it back up
- Heating involves putting an object in a microwave, turning the microwave on, turning it off, and then picking the object back up
- Cooling involves putting an object in a fridge, closing the fridge, opening it, and then picking the object back up
- Multiple actions can happen between frames, though MOST OFTEN it will just be a combination of goto and one other action. For example, it is unlikely we sliced AND put down the knife inbetween two frames.
- You are bound not to recognize some objects correctly, which might hurt your peformance in downstream tasks. Instead of running with your first guess, try to list a few alternatives for an object if you're unsure what it is.
- Whenever we hold a knife, we might be about to slice an object. Pay very close attention to minute details in objects' appearance. If there are small lines in the object where there previously weren't any, it is likely that we have sliced it. But it might have been already sliced, or maybe we just do not plan to slice it, even though we hold a knife.
- Sometimes, when putting down an object, we put it in or onto another object, instead of a container. In that case you should describe the object we put the object in or onto in addition to the general place this "container" is in.
- Similarly, sometimes we pick up an object that has another object in it. In that case, list both objects in your description.
"""


class Task:
    prefix: str
    name: str
    prompt_gpt: str
    example_gpt: str
    trajectories: list[Trajectory]
    descriptions: list[str]
    metadata: Optional[dict] = None

    def __init__(
        self,
        prefix,
        name,
        prompt_gpt,
        example_gpt,
        trajectories,
        metadata: Optional[dict] = None,
    ):
        self.prefix = prefix
        self.name = name
        self.trajectories = trajectories
        self.prompt_gpt = prompt_gpt
        self.example_gpt = example_gpt
        self.descriptions = []
        for t in trajectories:
            if t.description not in self.descriptions:
                self.descriptions.append(t.description)
        self.metadata = metadata
        self._check_classes()

    def _check_classes(self):
        if len(self.descriptions) < 2:
            raise ValueError(f"Not enough classes for a task: {self.descriptions}")
        if not all(t.description in self.descriptions for t in self.trajectories):
            raise ValueError("Class missing in descriptions.")

    @property
    def _used_concepts(self):
        objects = set(
            [a.object1 for t in self.trajectories for a in t.actions]
            + [a.object2 for t in self.trajectories for a in t.actions]
        )
        high_level_actions = set(
            [a.action for t in self.trajectories for a in t.actions]
        )
        low_level_actions = set(
            [
                lla.action
                for t in self.trajectories
                for hla in t.actions
                for lla in hla.actions
            ]
        )
        return {
            "objects": [o for o in objects if o],
            "high_level_actions": list(high_level_actions),
            "low_level_actions": list(low_level_actions),
        }

    def write(self, output_dir, max_videos=10, min_frames=32) -> str:
        tasks_dir = output_dir / "tasks" / "alfred" / self.prefix
        tasks_dir.mkdir(parents=True, exist_ok=True)
        videos_dir = output_dir / "videos" / "alfred" / self.prefix / self.name

        description_to_label = {
            d: f"label_{i}" for i, d in enumerate(self.descriptions)
        }
        label_to_description = {v: k for k, v in description_to_label.items()}

        # Task definition
        with open(tasks_dir / f"{self.name}.yaml", "w") as f:
            yaml.dump(
                {
                    "label_prompts": label_to_description,
                    "prompt_gpt": self.prompt_gpt,
                    "example_gpt": self.example_gpt,
                    "metadata": (self.metadata or {})
                    | {"concepts": self._used_concepts},
                },
                f,
            )

        videos = []
        video_counter = Counter()
        # Task videos
        # TODO Could be made more efficient by only having one copy of each video
        random.shuffle(self.trajectories)
        for trajectory in self.trajectories:
            if video_counter[trajectory.description] >= max_videos:
                continue
            video_counter[trajectory.description] += 1
            video_name = f"{trajectory.trial_id}_{uuid.uuid4()}.mp4"
            label = description_to_label[trajectory.description]
            (videos_dir / label).mkdir(parents=True, exist_ok=True)
            video = trajectory.video(min_frames=min_frames)
            video.write_videofile(str(videos_dir / label / video_name), logger=None)
            videos.append(
                {
                    "path": f"{self.prefix}/{self.name}/{label}/{video_name}",
                    "label": label,
                }
            )

        if len(set(video_counter.values())) != 1:
            print(f"WARNING: Unbalanced video classes: {video_counter}")

        # Task data
        with open(tasks_dir / f"{self.name}_data.json", "w") as f:
            json.dump(videos, f)

        return f"{self.prefix}/{self.name}"


def write_config(tasks: list[Task], output_dir: Path):
    # Config
    with open(output_dir / "configs" / "alfred.yaml", "w") as f:
        config = {
            "tasks": tasks,
            "models": [
                {
                    "kind": "gpt",
                    "model": "gpt-4o",
                    "async_batch": False,
                    "n_frames": 16,
                    "is_one_shot": True,
                },
                {
                    "kind": "gpt",
                    "model": "gpt-4o",
                    "async_batch": False,
                    "n_frames": 16,
                    "is_one_shot": False,
                },
                {
                    "kind": "gpt",
                    "model": "gpt-4o",
                    "async_batch": False,
                    "n_frames": 5,
                    "is_one_shot": False,
                },
                {"kind": "encoder", "encoder": "s3d", "heads": [{"kind": "cosine"}]},
                {"kind": "encoder", "encoder": "viclip", "heads": [{"kind": "cosine"}]},
                {
                    "kind": "encoder",
                    "encoder": "clip",
                    "heads": [{"kind": "cosine"}],
                    "hf_model": "ViT-bigG-14/laion2b_s39b_b160k",
                    "n_frames": 4,
                },
                {
                    "kind": "encoder",
                    "encoder": "clip",
                    "heads": [{"kind": "cosine"}],
                    "hf_model": "ViT-bigG-14/laion2b_s39b_b160k",
                    "n_frames": 32,
                    "batch_size": 1,
                },
            ],
            "task_dir": "tasks/alfred",
            "video_dir": "videos/alfred",
            "cache_dir": ".cache",
            "output_dir": "experiments",
        }
        yaml.dump(config, f)


def get_clipped_trajectories(
    trajectories, action, context=5
) -> list[tuple[int, Trajectory]]:
    result = []
    for t in trajectories:
        for idx in [i for i, a in enumerate(t.actions) if a.action == action]:
            hl_actions = deepcopy(t.actions)
            hl_actions[idx - 1].actions = hl_actions[idx - 1].actions[-context:]
            if len(hl_actions) > idx + 1:
                hl_actions[idx + 1].actions = hl_actions[idx + 1].actions[:context]
            beg = (
                idx - 1
                if idx > 0 and hl_actions[idx - 1].action == "GotoLocation"
                else idx
            )
            end = (
                idx + 2
                if idx + 1 < len(hl_actions)
                and hl_actions[idx + 1].action == "GotoLocation"
                else idx + 1
            )
            new_t = t.with_modified_actions(hl_actions[beg:end])
            new_t._description = hl_actions[idx].description.capitalize()
            result.append((idx - beg, new_t))
    return result


def group_trajectories(
    trajectories: list[tuple[int, Trajectory]],
    object1: bool = False,
    object2: bool = False,
) -> list[tuple[str | list[str], str | list[str | None] | None, list[Trajectory]]]:
    return [
        (
            (
                g[0][1].actions[g[0][0]].object1
                if object1
                else [t.actions[idx].object1 for idx, t in g]
            ),
            (
                g[0][1].actions[g[0][0]].object2
                if object2
                else [t.actions[idx].object2 for idx, t in g]
            ),
            [t for _, t in g],
        )
        for g in utils.group_by(
            trajectories,
            lambda t: (
                not object1 or t[1].actions[t[0]].object1,
                not object2 or t[1].actions[t[0]].object2,
            ),
        )
    ]


def prompt_object(container):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict us or putting down an object into (or on) a {container}. Your eventual goal will be to recognize the object, but you shouldn't lock-in to one answer too early. Instead, try to describe the object as accurately as possible separately for each frame, refining your answer as you see more frames.
""".strip()


def example_object(container):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the very bottom of the screen and close to the camera, without any visible hands. Similarly, no hands are shown for cleaning, heating, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen, close to the camera and looks like it is lying on the floor, we are most likely just holding it.
- You are bound not to recognize some objects correctly, which might hurt your peformance in downstream tasks. Instead of running with your first guess, try to list a few alternatives for an object if you're unsure what it is.
- If we didn't have anything in hand in a frame, and something appears in our hands in the next frame, we must have performed a `pick` action

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are in the middle of the room. We do not see the {container} yet.
2. The location has changed. We see something that might be the {container}. We see a multitude of objects there, among them a toaster, a butter knife, a laptop, and a mug, and a pan or possibly a bowl.
2. The scene has not changed much since the last frame. We are still near the {container}.
3. We see a rectangular object at the bottom of the screen, suggesting we are holding it. Considering the selection of objects on the {container} we saw above, it is likely the laptop or the toaster.
4. The angle has changed a bit and we see that the laptop is still on the {container}. This means the thing we hold must be the toaster, based on our previous observations.
5. We are now a bit further away from the {container}. We are still holding the toaster.
""".strip()


def object_tasks(trajectories: list[Trajectory]) -> list[Task]:
    all_objects = set(plausible_action_objects["PickupObject"][0])
    object_trajectories = get_clipped_trajectories(trajectories, "PickupObject")

    seen = set()
    tasks = []
    for objects, container, g in group_trajectories(object_trajectories, object2=True):
        assert (container is None or isinstance(container, str)) and isinstance(
            objects, list
        )
        if seen == objects:
            break
        container_trajectories = [
            t for object, t in zip(objects, g) if object not in seen
        ]
        if not container_trajectories:
            continue
        try:
            name = (container or "somewhere").replace(" ", "_")
            tasks.append(
                Task(
                    prefix="foundation/objects",
                    name=f"pick_from_{name}",
                    trajectories=container_trajectories,
                    prompt_gpt=prompt_object(container),
                    example_gpt=example_object(container),
                )
            )
            seen |= set(objects)
        except ValueError as e:
            print(f"Skipping object {container} due to problems: {e}")

    if seen != objects:
        print(f"Missing containers: {all_objects - seen}")

    return tasks


def prompt_container(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict us putting down a {object} somewhere. Your eventual goal will be to recognize the place or container we put the object in (or on), but you shouldn't lock-in to one answer too early. Instead, try to describe the place or container as accurately as possible separately for each frame, refining your answer as you see more frames.
""".strip()


def example_container(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the very bottom of the screen and close to the camera, without any visible hands. Similarly, no hands are shown for cleaning, heating, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen, close to the camera and looks like it is lying on the floor, we are most likely just holding it.
- You are bound not to recognize some objects correctly, which might hurt your peformance in downstream tasks. Instead of running with your first guess, try to list a few alternatives for an object if you're unsure what it is.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are in the middle of the room. We do not see any {object} yet.
2. The location has changed. We see something that might be the {object}. It seems to be in a pot which itself is on a sofa.
2. The scene has not changed much since the last frame. We are still near the sofa. The {object} is in the bottom of the frame and close to the camera, suggesting we are holding it now. Considering the descriptions above, it is likely that the {object} was picked up from a pot, or possibly from a sofa.
4. Not much has changed, we are still near the sofa, and we still hold the {object}.
5. We are now a bit further away from the sofa. We are still holding the {object}.
""".strip()


def container_tasks(trajectories: list[Trajectory]) -> list[Task]:
    all_containers = set(plausible_action_objects["PutObject"][1])
    container_trajectories = get_clipped_trajectories(trajectories, "PutObject")
    seen = set()
    tasks = []
    for object, containers, g in group_trajectories(
        container_trajectories, object1=True
    ):
        assert isinstance(object, str) and isinstance(containers, list)
        if seen == all_containers:
            break
        obj_trajectories = [
            t for container, t in zip(containers, g) if container not in seen
        ]
        if not obj_trajectories:
            continue
        if not object:
            object = "somewhere"
        try:
            tasks.append(
                Task(
                    prefix="foundation/containers",
                    name=f"place_{object.replace(' ', '_')}",
                    trajectories=obj_trajectories,
                    prompt_gpt=prompt_container(object),
                    example_gpt=example_container(object),
                )
            )
            seen |= set(containers)
        except ValueError as e:
            print(f"Skipping object {object} due to problems: {e}")

    if seen != all_containers:
        print(f"Missing containers: {all_containers - seen}")

    return tasks


def prompt_cleaning(object):
    return f"""
The frames will depict us carrying a {object} to a sink and then possibly cleaning it under running water. Your eventual goal will be to discern whether all the steps required for cleaning the object actually happened, but you shouldn't lock-in to one answer too early. Instead, for each frame, concisely describe the scene, the objects we're holding, and from that deduce what actions have likely been performed since the last frame.
""".strip()


def example_cleaning(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the very bottom of the screen and close to the camera, without any visible hands. Similarly, no hands are shown for cleaning, heating, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen, close to the camera and looks like it is lying on the floor, we are most likely just holding it.
- Proper cleaning involves putting an object in a sink, running water over it, and then picking it back up

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are near the countertop. There is a {object} at the bottom of the screen, which suggests we are holding it.
2. We are now near a sink. The {object} is still at the bottom of the screen, so we are definitely holding it.
3. The {object} now lies in the sink, along with other objects. The water is not running yet.
4. The {object} is still in the sink. The water is still not running.
5. We are holding the {object} again, as depicted by it being at the bottom of the screen again. Although it was in the sink, we didn't run water over it, which means we did not clean it.
"""


def clean_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []
    cleaning_trajectories = get_clipped_trajectories(trajectories, "CleanObject")
    groups = group_trajectories(cleaning_trajectories, object1=True, object2=True)
    for object, container, g in groups:
        assert isinstance(object, str) and isinstance(container, str)
        task_trajectories = g.copy()

        # Go to, Go away
        for t in g:
            idx = [a.action for a in t.actions].index("CleanObject")
            alt_actions = deepcopy(t.actions)
            alt_t = t.with_modified_actions(alt_actions[:idx] + alt_actions[idx + 1 :])
            alt_t._description = f"We go to the {container}, holding {object} in hand, we don't put it in the {container}"
            task_trajectories.append(alt_t)

        # Go to, Put, Pick, Go away
        for t in g:
            alt_t = t.filter_low_level(lambda lla: "ToggleObject" not in lla.action)
            alt_t._description = f"We put the {object} in the {container} and pick it back up without running water over it"
            task_trajectories.append(alt_t)

        name = f"{object.replace(' ', '_')}"
        tasks.append(
            Task(
                prefix="foundation/clean",
                name=name,
                trajectories=task_trajectories,
                prompt_gpt=prompt_cleaning(object),
                example_gpt=example_cleaning(object),
            )
        )
    return tasks


def prompt_heating(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict us carrying a {object} to a microwave and then possibly heating it there. Your eventual goal will be to discern whether all the steps required for heating the object actually happened, but you shouldn't lock-in to one answer too early. Instead, for each frame, concisely describe the scene, the objects we're holding, and what actions have likely been performed since the last frame.
""".strip()


def example_heating(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the very bottom of the screen and close to the camera, without any visible hands. Similarly, no hands are shown for cleaning, heating, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen, close to the camera and looks like it is lying on the floor, we are most likely just holding it.
- Proper heating involves putting an object in a microwave, closing the door, turning the microwave on, turning it off, opening the door and then picking the object back up
- When the microwave is turned on, it lights up

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are near the countertop. There is a {object} at the bottom of the screen, which suggests we are holding it.
2. We are now near a microwave. The {object} is still at the bottom of the screen, so we are still holding it.
3. The {object} now lies in the microwave. The microwave is open and thus not running, so we are not heating the {object} up yet.
4. The {object} is still in the microwave. The microwave is still open.
5. We are holding the {object} again, as depicted by it being at the bottom of the screen again. Although it was in the microwave, we didn't close the microwave and turn it on, which means we did not clean it.
"""


def heat_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []
    heating_trajectories = get_clipped_trajectories(trajectories, "HeatObject")
    groups = group_trajectories(heating_trajectories, object1=True, object2=True)
    for object, container, g in groups:
        assert isinstance(object, str) and isinstance(container, str)
        task_trajectories = g.copy()

        sequence = [
            "OpenObject",
            "PutObject",
            "CloseObject",
            "ToggleObjectOn",
            "ToggleObjectOff",
            "OpenObject",
            "PickupObject",
            "CloseObject",
        ]

        # Go to, Go away
        for t in g:
            alt_actions = deepcopy(t.actions)
            idx = [a.action for a in t.actions].index("HeatObject")
            alt_t = t.with_modified_actions(alt_actions[:idx] + alt_actions[idx + 1 :])
            alt_t._description = f"We go to the {container} and then immediately leave without putting the {object} in"
            task_trajectories.append(alt_t)

        # Go to, Open, Close, Go away
        for t in g:
            alt_actions = deepcopy(t.actions)
            idx = [a.action for a in t.actions].index("HeatObject")
            hla = alt_actions[idx]
            llas = hla.actions
            if [lla.action for lla in llas] != sequence:
                continue
            hla.actions = llas[:1] + llas[7:]
            alt_t = t.with_modified_actions(alt_actions)
            alt_t._description = f"We open the {container} and immediately close it again, without putting the {object} in"
            task_trajectories.append(alt_t)

        # Go to, Open, Put, Pick up, Close, Go away
        for t in g:
            alt_actions = deepcopy(t.actions)
            idx = [a.action for a in t.actions].index("HeatObject")
            hla = alt_actions[idx]
            llas = hla.actions
            if [lla.action for lla in llas] != sequence:
                continue
            hla.actions = llas[:2] + llas[6:]
            alt_t = t.with_modified_actions(alt_actions)
            alt_t._description = f"We open the {container}, put the {object} in, then immediately pick it back up without heating it"
            task_trajectories.append(alt_t)

        # Go to, Open, Put, Close, Open, Pick up, Close, Go away
        for t in g:
            alt_t = t.filter_low_level(lambda lla: "ToggleObject" not in lla.action)
            alt_t._description = f"We put the {object} in the {container} for a while, we do not turn the {container} on, then we pick up the {object} back again"
            task_trajectories.append(alt_t)

        name = f"{object.replace(' ', '_')}"
        tasks.append(
            Task(
                prefix="foundation/heat",
                name=name,
                trajectories=task_trajectories,
                prompt_gpt=prompt_heating(object),
                example_gpt=example_heating(object),
            )
        )
    return tasks


def prompt_cooling(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict us carrying a {object} to a fridge and then possibly cooling it by leaving it in a closed fridge for a while before picking it up again. Your eventual goal will be to discern whether all the steps required for cooling the object actually happened, but you shouldn't lock-in to one answer too early. Instead, for each frame, concisely describe the scene, the objects we're holding, and what actions have likely been performed since the last frame.
""".strip()


def example_cooling(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the very bottom of the screen and close to the camera, without any visible hands. Similarly, no hands are shown for cleaning, heating, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen, close to the camera and looks like it is lying on the floor, we are most likely just holding it.
- Cooling an object involves holding it, opening the fridge, putting it in a fridge, closing the fridge, opening the fridge, and picking the object back up. Proper cooling needs to have all these parts.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are near the countertop. There is {object} at the bottom of the screen, which suggests we are holding it.
2. We are now near a fridge. The {object} is still at the bottom of the screen, so we are definitely holding it.
3. The fridge is now open. We still hold the {object}.
4. The fridge is now closed. We still hold the {object}. This means we did not put the object in the fridge.
5. We are holding the {object}, still. It seems we just opened and closed the fridge without putting the {object} in, even for a moment. The cooling did not happen.
"""


def cool_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []
    cooling_trajectories = get_clipped_trajectories(trajectories, "CoolObject")
    groups = group_trajectories(cooling_trajectories, object1=True, object2=True)
    for object, container, g in groups:
        assert isinstance(object, str) and isinstance(container, str)
        task_trajectories = g.copy()

        sequence = [
            "OpenObject",
            "PutObject",
            "CloseObject",
            "OpenObject",
            "PickupObject",
            "CloseObject",
        ]

        # Go to, Open, Put, Immediately pick up, Close, Go away
        for t in g:
            alt_actions = deepcopy(t.actions)
            idx = [a.action for a in t.actions].index("CoolObject")
            hla = alt_actions[idx]
            llas = hla.actions
            if [lla.action for lla in llas] != sequence:
                continue
            hla.actions = llas[:2] + llas[4:]
            alt_t = t.with_modified_actions(alt_actions)
            alt_t._description = f"We open the {container}, put in the {object} and immediately pick it back up without leaving it to cool in a closed {container}"
            task_trajectories.append(alt_t)

        # Go to, Open, Close, Go away
        for t in g:
            alt_actions = deepcopy(t.actions)
            idx = [a.action for a in t.actions].index("CoolObject")
            hla = alt_actions[idx]
            llas = hla.actions
            if [lla.action for lla in llas] != sequence:
                continue
            hla.actions = llas[:1] + llas[5:]
            alt_t = t.with_modified_actions(alt_actions)
            alt_t._description = f"We open the {container} and then immediately close it without putting the {object} in"
            task_trajectories.append(alt_t)

        # Go to, Go away
        for t in g:
            idx = [a.action for a in t.actions].index("CoolObject")
            alt_actions = t.actions[:idx] + t.actions[idx + 1 :]
            alt_t = t.with_modified_actions(alt_actions)
            alt_t._description = (
                f"We go to the {container} and then we leave, without even opening it"
            )
            task_trajectories.append(alt_t)

        name = f"{object.replace(' ', '_')}"
        tasks.append(
            Task(
                prefix="foundation/cool",
                name=name,
                trajectories=task_trajectories,
                prompt_gpt=prompt_cooling(object),
                example_gpt=example_cooling(object),
            )
        )
    return tasks


def prompt_toggle(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict a {object}, and your eventual goal will be to discern whether we turned it on or off during the video. You shouldn't lock-in to one answer too early, though — instead, for each frame, concisely describe the scene and the state of the {object} and what actions have likely been performed since the last frame.
""".strip()


def example_toggle(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the very bottom of the screen and close to the camera, without any visible hands. Similarly, no hands are shown for cleaning, heating, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen, close to the camera and looks like it is lying on the floor, we are most likely just holding it.
- We might be carrying an object in the frames, but what we care about the most is turning other objects on or off.
- The state of the {object} will change exactly once in the frames, from on to off or vice versa.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are near the countertop. We see a {object} in front of us.
2. The {object} doesn't seem to be turned on.
3. Nothing has changed; the {object} is still off.
4. Now, the {object} is turned on. We can see this because it the light inside has turned on.
5. The {object} is still on.
"""


def toggle_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []

    toggling_trajectories = []
    for t in trajectories:
        toggling_indices = [
            (i, j)
            for i, hla in enumerate(t.actions)
            for j, lla in enumerate(hla.actions)
            if lla.action == "ToggleObjectOn"
        ]
        for hla_idx, lla_idx in toggling_indices:
            hl_actions = deepcopy(t.actions)
            hl_actions[hla_idx - 1].actions = hl_actions[hla_idx - 1].actions[-3:]
            beg = (
                hla_idx - 1
                if hla_idx > 0 and hl_actions[hla_idx - 1].action == "GotoLocation"
                else hla_idx
            )
            end = hla_idx + 1
            toggle_action = hl_actions[hla_idx]
            toggle_action.actions = [toggle_action.actions[lla_idx]]
            new_t = t.with_modified_actions(hl_actions[beg:end])
            new_t._description = f"We turn the {toggle_action.actions[0].object1} on"
            toggling_trajectories.append((hla_idx - beg, new_t))

    for g in utils.group_by(
        toggling_trajectories, lambda t: t[1].actions[t[0]].actions[0].object1
    ):
        object, g = g[0][1].actions[g[0][0]].actions[0].object1, [t for _, t in g]

        task_trajectories = g.copy() + [
            t.reverse(f"We turn the {object} off") for t in g
        ]

        name = f"{object.replace(' ', '_')}"
        tasks.append(
            Task(
                prefix="foundation/toggle",
                name=name,
                trajectories=task_trajectories,
                prompt_gpt=prompt_toggle(object),
                example_gpt=example_toggle(object),
            )
        )

    return tasks


def prompt_pick(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict us handling a {object}, and your eventual goal will be to discern whether picked it up during the video, or started out holding it and put it down somewhere. You shouldn't lock-in to one answer too early. Instead, for each frame, concisely describe the scene, the state of the {object}, and what actions have likely been performed since the last frame.
""".strip()


def example_pick(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the very bottom of the screen and close to the camera, without any visible hands. Similarly, no hands are shown for cleaning, heating, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen, close to the camera and looks like it is lying on the floor, we are most likely just holding it.
- Sometimes it is hard to see where exactly we put the object. However, if we held it in a frame and we don't hold it in the next frame, we definitely had to put it somewhere.
- We will never pick up an object and put it down in the same frame. Only exactly one of these will happen.
- Sometimes, when putting down an object, we put it in or onto another object, instead of a container. In that case you should describe the object we put the object in or onto in addition to the general place this "container" is in.
- Similarly, sometimes we pick up an object that has another object in it. In that case, list both objects in your description.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are near a countertop or a large table.
2. Now it seems we walked near a windowsill. We don't appear to have anything in hand. There is a {object} on the windowsill, along with a toaster, and a bowl or a plate.
3. We seem to be holding the {object} now, because it is at the bottom of the screen, almost as if it was lying on the floor.
4. We still hold the {object}. We are further from the windowsill now, possibly walking towards somewhere else.
5. We still hold the {object}.
"""


def pick_up_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []
    picking_trajectories = get_clipped_trajectories(trajectories, "PickupObject")
    putting_trajectories = get_clipped_trajectories(trajectories, "PutObject")
    groups = group_trajectories(
        picking_trajectories + putting_trajectories, object1=True, object2=True
    )
    for object, container, g in groups:
        if container is None or not set(
            a.action for t in g for a in t.actions
        ).issuperset({"PickupObject", "PutObject"}):
            continue
        assert isinstance(object, str) and isinstance(container, str)

        name = f"{object.replace(' ', '_')}_{container.replace(' ', '_')}"
        tasks.append(
            Task(
                prefix="foundation/pick_v_put",
                name=name,
                trajectories=g,
                prompt_gpt=prompt_pick(object),
                example_gpt=example_pick(object),
            )
        )

    return tasks


def slicing_prompt(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict us carrying a knife and walking towards a {object}. Your eventual goal will be to discern whether we sliced the {object} at the end of the video or not, but you shouldn't lock-in to one answer too early. Instead, for each frame, concisely describe the scene, the state of the {object}, and what actions have likely been performed since the last frame.
""".strip()


def example_slicing(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the very bottom of the screen and close to the camera, without any visible hands. Similarly, no hands are shown for cleaning, heating, or any other action, not even slicing.
- The objects almost never lie on the floor. If the object is at the bottom of the screen, close to the camera and looks like it is lying on the floor, we are most likely just holding it.
- Sometimes we just walk to an object with a knife in hand without slicing the object. To see whether an object has been sliced, first find it in the frame, and then observe how it changes in the following frames. A sliced object will have small, barely noticeable lines on it.
- Whenever we hold a knife, we might be about to slice an object. Pay very close attention to minute details in objects' appearance. If there are small lines in the object where there previously weren't any, it is likely that we have sliced it. But it might have been already sliced, or maybe we just do not plan to slice it, even though we hold a knife.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are near the countertop. A butter knife is shown on the bottom of the screen, which means we are holding it, as expected.
2. We have walked near a windowsill, upon which there is a multitude of objects, including {object}. The knife is still at the bottom of the screen, so we are definitely holding it.
3. We are holding the knife still, but it is not clear if we are using it. Focusing on the {object}, it seems unchanged.
4. The {object} now has lines on it, which suggests we have sliced it. We still hold the knife, and the {object} is still in the same place.
5. We are still near the windowsill. We now see the {object} from a different angle, and although we now do not see the slicing lines, we did slice it according to the previous frame.
"""


def slice_tasks(trajectories: list[Trajectory]) -> list[Task]:
    slicing_trajectories = get_clipped_trajectories(trajectories, "SliceObject")
    tasks = []
    for object, _, g in group_trajectories(slicing_trajectories, object1=True):
        assert isinstance(object, str)
        task_trajectories = g.copy()
        for t in g:
            idx = [a.action for a in t.actions].index("SliceObject")
            if idx == 0:
                continue
            alt_t = t[:idx]
            alt_t._description = (
                f"We walk to the {object} with a knife in hand but don't slice it"
            )
            task_trajectories.append(alt_t)

        tasks.append(
            Task(
                prefix="foundation/slice",
                name=f"{object.replace(' ', '_')}",
                trajectories=task_trajectories,
                prompt_gpt=slicing_prompt(object),
                example_gpt=example_slicing(object),
            )
        )
    return tasks


def sliced_v_whole_prompt(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict us picking up a {object}, or possibly just a piece or slice of it. Your eventual goal will be to say whether the object was picked up whole, or if just a piece of it was picked, but you shouldn't lock-in to one answer too early. Instead, for each frame, concisely describe the scene, the state of the {object}, and what actions have likely been performed since the last frame.
""".strip()


def example_sliced_v_whole(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the very bottom of the screen and close to the camera, without any visible hands. Similarly, no hands are shown for cleaning, heating, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen, close to the camera and looks like it is lying on the floor, we are most likely just holding it.
- We are sure that you will observe a {object} being picked up. If you can barely see it, maybe you just picked up a very thin slice that is angled weirdly.
- Sometimes we will pick up slices, sometimes wedges, but you can usually ignore this, as long as you can tell whether the object is whole or not.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are nearing a countertop. We see a {object} lying on the countertop. It is hard to see whether it is sliced or not when it's lying down like this.
2. We are now standing directly at the countertop.
3. We picked up the {object}. It seems to be a wedge instead of the whole thing.
4. We now changed our position again. We can now see the potato is just a wedge since we can see the inner texture.
5. We are now nearing a fridge, still holding the {object}, as signified by it being at the bottom of the screen.
"""


def sliced_v_whole_tasks(trajectories: list[Trajectory]) -> list[Task]:
    picking_trajectories = get_clipped_trajectories(trajectories, "PickupObject")
    groups = utils.group_by(
        picking_trajectories,
        lambda t: (t[1].actions[t[0]].object1.removeprefix("slice of ")),
    )
    tasks = []
    for g in groups:
        if len(set(t.actions[idx].object1 for idx, t in g)) == 1:
            # Skip, we are only interested with actions where the object is both sliced and non-sliced
            continue
        object = g[0][1].actions[g[0][0]].object1.removeprefix("slice of ")
        # container = g[0][1].actions[g[0][0]].object2 or "somewhere"
        task_trajectories = []
        for idx, t in g:
            t._description = f"We pick up a {t.actions[idx].object1}"
            task_trajectories.append(t)
        tasks.append(
            Task(
                prefix="foundation/sliced_v_whole",
                name=object.replace(" ", "_"),
                trajectories=task_trajectories,
                prompt_gpt=sliced_v_whole_prompt(object),
                example_gpt=example_sliced_v_whole(object),
            )
        )
    return tasks


def prompt_on_v_off(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will standing in front of or walking towards a {object}. Your eventual goal will be to say whether the {object} has been on or off in the video. The on/off state of the object won't change during the video. You shouldn't lock-in to one answer too early — instead, for each frame, concisely describe the scene, the state of the {object}, and what actions have likely been performed since the last frame.
""".strip()


def example_on_v_off(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the very bottom of the screen and close to the camera, without any visible hands. Similarly, no hands are shown for cleaning, heating, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen, close to the camera and looks like it is lying on the floor, we are most likely just holding it.
- Sometimes the {object} will be barely visible in the frame. Still, you should be able to discern whether it is shining (and thus on) or not.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are near the countertop. We see a {object} in front of us.
2. The {object} doesn't seem to be turned on. There is no light coming from it.
3. Nothing has changed; the {object} is still off.
4. Again, similar frame, the {object} is still off, as we would expect.
5. For the entire length of the video, the {object} has been off.
"""


def on_v_off_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []

    toggling_trajectories = []
    for t in trajectories:
        toggling_indices = [
            (i, j)
            for i, hla in enumerate(t.actions)
            for j, lla in enumerate(hla.actions)
            if lla.action == "ToggleObjectOn"
        ]
        for hla_idx, lla_idx in toggling_indices:
            hl_actions = deepcopy(t.actions)
            hl_actions[hla_idx - 1].actions = hl_actions[hla_idx - 1].actions[-5:]
            if len(hl_actions) > hla_idx + 1:
                hl_actions[hla_idx + 1].actions = hl_actions[hla_idx + 1].actions[:5]
            beg = (
                hla_idx - 1
                if hla_idx > 0 and hl_actions[hla_idx - 1].action == "GotoLocation"
                else hla_idx
            )
            end = (
                hla_idx + 2
                if hla_idx + 1 < len(hl_actions)
                and hl_actions[hla_idx + 1].action == "GotoLocation"
                else hla_idx + 1
            )
            toggle_action = hl_actions[hla_idx]
            toggle_action.actions = [toggle_action.actions[lla_idx]]
            new_t = t.with_modified_actions(hl_actions[beg:end])
            toggling_trajectories.append((hla_idx - beg, new_t))

    for g in utils.group_by(
        toggling_trajectories, lambda t: t[1].actions[t[0]].actions[0].object1
    ):
        object = g[0][1].actions[g[0][0]].actions[0].object1

        task_trajectories = []
        for idx, t in g:
            t_on, t_off = deepcopy(t), deepcopy(t)
            t_on.actions = [t_on.actions[idx]]
            t_on.actions[0].actions[0].images = t.actions[idx].actions[0].images[-3:]
            t_on._description = f"The {object} is turned on"
            t_off.actions[idx].actions[0].images = t.actions[idx].actions[0].images[:3]
            t_off._description = f"The {object} is turned off"
            task_trajectories += [t_on, t_off]

        name = f"{object.replace(' ', '_')}"
        tasks.append(
            Task(
                prefix="foundation/on_v_off",
                name=name,
                trajectories=task_trajectories,
                prompt_gpt=prompt_on_v_off(object),
                example_gpt=example_on_v_off(object),
            )
        )

    return tasks


def actions_to_blocks(actions: list[HighLevelAction]) -> list[list[HighLevelAction]]:
    grouped_actions, group = [], []
    for a in actions:
        if group and group[-1].action != "GotoLocation":
            grouped_actions.append(group)
            group = []
        group.append(a)
    if group:
        grouped_actions.append(group)
    return grouped_actions


def from_action_blocks(blocks: list[list[HighLevelAction]]) -> list[HighLevelAction]:
    return [a for block in blocks for a in block]


def equivalent_trajectory_present(
    trajectory: Trajectory, trajectories: list[Trajectory]
):
    actions = [a for a in trajectory.actions if a.action != "GotoLocation"]
    for t in trajectories:
        alt_actions = [a for a in t.actions if a.action != "GotoLocation"]
        if all(a1 == a2 for a1, a2 in zip(actions, alt_actions)):
            return True
    return False


def find_next_block(
    current: list[list[HighLevelAction]],
    all_blocks: dict[tuple[str, str | None], list[list[HighLevelAction]]],
    all_gotos: dict[
        str, dict[tuple[str | None, str | None, str | None], list[HighLevelAction]]
    ],
) -> Generator[list[HighLevelAction], Any, None]:
    if not current:
        yield from (b for l in all_blocks.values() for b in l)
    else:
        last = current[-1][-1]
        blocks = all_blocks.get((last.scene, last.end_hand_content), [])
        random.shuffle(blocks)
        blocks = sorted(
            blocks, key=lambda b: block_to_action(b).action != "ToggleObject"
        )

        for next_block in blocks:
            next_location = next_block[0].beg_location
            if next_location == last.end_location:
                yield next_block
            else:
                if gotos := all_gotos.get(last.scene, {}).get(
                    (last.end_location, next_location, last.end_hand_content), []
                ):
                    yield [random.choice(gotos)] + next_block


def find_remaining_blocks(
    current: list[list[HighLevelAction]],
    target_len: int,
    all_blocks: dict[tuple[str, str | None], list[list[HighLevelAction]]],
    all_gotos: dict[
        str, dict[tuple[str | None, str | None, str | None], list[HighLevelAction]]
    ],
) -> list[list[HighLevelAction]] | None:
    assert len(current) > 0

    if len(current) == target_len:
        return current

    for next_block in find_next_block(current, all_blocks, all_gotos):
        action = block_to_action(next_block)
        if any(action == other_a for block in current for other_a in block):
            continue

        return find_remaining_blocks(
            current=current + [next_block],
            target_len=target_len,
            all_blocks=all_blocks,
            all_gotos=all_gotos,
        )

    return None


def compute_blocks(
    trajectories: list[Trajectory],
) -> dict[tuple[str, str | None], list[list[HighLevelAction]]]:
    all_blocks: dict[tuple[str, str | None], list[list[HighLevelAction]]] = {}
    for trajectory in trajectories:
        for g in actions_to_blocks(trajectory.actions):
            scene, hand_content = g[0].scene, g[0].beg_hand_content
            all_blocks.setdefault((scene, hand_content), []).append(g)
    return all_blocks


def compute_gotos(
    trajectories: list[Trajectory],
) -> dict[str, dict[tuple[str | None, str | None, str | None], list[HighLevelAction]]]:
    goto_actions: dict[
        str, dict[tuple[str | None, str | None, str | None], list[HighLevelAction]]
    ] = {}
    for trajectory in trajectories:
        for a in [a for a in trajectory.actions if a.action == "GotoLocation"]:
            scene_gotos = goto_actions.setdefault(a.scene, {})
            matching_actions = scene_gotos.setdefault(
                (a.beg_location, a.end_location, a.beg_hand_content), []
            )
            matching_actions.append(a)
    return goto_actions


def block_to_action(block: list[HighLevelAction]) -> HighLevelAction:
    return next(a for a in block if a.action != "GotoLocation")


def remixed_task(
    seed: Trajectory,
    videos_on_prefix_0: int,
    videos_per_prefix: int,
    all_trajectories: list[Trajectory],
) -> Task:
    all_blocks = compute_blocks(all_trajectories)
    all_gotos = compute_gotos(all_trajectories)
    seed_blocks = actions_to_blocks(seed.actions)
    target_len = len(seed_blocks)

    trajectories = {target_len: [seed]}
    for prefix_len in range(target_len - 1, -1, -1):
        prefix = seed_blocks[:prefix_len]
        target_count = videos_on_prefix_0 if prefix_len == 0 else videos_per_prefix
        for next_block in find_next_block(prefix, all_blocks, all_gotos):
            # We have enough trajectories for this prefix length
            if len(trajectories.get(prefix_len, [])) == target_count:
                break
            flat_trajectories = [t for l in trajectories.values() for t in l]
            # The first step in the trajectory has to be different from others
            if any(
                block_to_action(next_block)
                == block_to_action(actions_to_blocks(t.actions)[prefix_len])
                for t in flat_trajectories
            ):
                continue

            blocks = find_remaining_blocks(
                prefix + [next_block], target_len, all_blocks, all_gotos
            )
            if blocks is None:
                continue
            trajectory = seed.with_modified_actions(from_action_blocks(blocks))
            assert not equivalent_trajectory_present(trajectory, flat_trajectories)
            trajectories.setdefault(prefix_len, []).append(trajectory)

    return Task(
        prefix=f"level_{target_len}/remix",
        name=str(uuid.uuid4()),
        trajectories=[t for l in trajectories.values() for t in l],
        metadata={
            "prefix_lens": [
                {"shared_prefix": prefix_len, "description": t.description}
                for prefix_len, trajectories in trajectories.items()
                for t in trajectories
            ]
        },
        prompt_gpt=HIGH_LEVEL_TASK_PROMPT,
        example_gpt=HIGH_LEVEL_TASK_EXAMPLE,
    )


def try_make_consistent(
    actions: list[HighLevelAction],
    all_goto: dict[
        str, dict[tuple[str | None, str | None, str | None], list[HighLevelAction]]
    ],
) -> list[HighLevelAction] | None:
    result = [actions[0]]
    for prev, next in zip(actions, actions[1:]):
        if prev.end_hand_content != next.beg_hand_content:
            return None

        if prev.end_location != next.beg_location:
            gotos = all_goto.get(prev.scene, {}).get(
                (prev.end_location, next.beg_location, prev.end_hand_content), []
            )
            if not gotos:
                return None
            result += [gotos[0], next]
        else:
            result.append(next)
    return result


def slicing_consistent(actions: list[HighLevelAction]) -> bool:
    for slicing_i in [i for i, a in enumerate(actions) if a.action == "SliceObject"]:
        sliced_object = actions[slicing_i].object1
        whole_before = all(
            i <= slicing_i
            for i, a in enumerate(actions)
            if (sliced_object in a.description and "slice" not in a.description)
        )
        sliced_after = all(
            i >= slicing_i
            for i, a in enumerate(actions)
            if ("slice" in a.description and sliced_object in a.description)
        )
        if not (whole_before and sliced_after):
            return False
    return True


def permutation_task(
    seed: Trajectory,
    num_classes: int,
    trajectories: list[Trajectory],
) -> Task:
    goto_actions = compute_gotos(trajectories)
    important_actions = [a for a in seed.actions if a.action != "GotoLocation"]

    consistent_trajectories, inconsistent_trajectories = [seed], []
    for perm in itertools.permutations(important_actions):
        if len(consistent_trajectories) == num_classes:
            break

        new_actions = try_make_consistent(list(perm), goto_actions)
        if new_actions is None:
            continue

        new_trajectory = seed.with_modified_actions(new_actions)

        if not equivalent_trajectory_present(
            new_trajectory, consistent_trajectories + inconsistent_trajectories
        ):
            if slicing_consistent(list(perm)):
                consistent_trajectories.append(new_trajectory)
            else:
                inconsistent_trajectories.append(new_trajectory)

    task_trajectories = (consistent_trajectories + inconsistent_trajectories)[
        :num_classes
    ]

    return Task(
        prefix=f"level_{len(important_actions)}/permutation",
        name=str(uuid.uuid4()),
        trajectories=task_trajectories,
        prompt_gpt=HIGH_LEVEL_TASK_PROMPT,
        example_gpt=HIGH_LEVEL_TASK_EXAMPLE,
        metadata={
            "is_consistent_wrt_slicing": len(consistent_trajectories) == num_classes,
        },
    )
