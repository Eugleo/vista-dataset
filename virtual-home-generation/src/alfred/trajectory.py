import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from alfred import object


@dataclass
class LowLevelAction:
    images: list[int]
    action: str
    object1: Optional[str] = None
    object2: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.action, self.object1, self.object2))

    def __eq__(self, other) -> bool:
        return (
            self.action == other.action
            and self.object1 == other.object1
            # If the 2nd object is None, it matches everything
            and (not self.object2 or not other.object2 or self.object2 == other.object2)
        )

    @staticmethod
    def from_json(js: dict, images: list[int]) -> "LowLevelAction":
        action = js["api_action"]["action"]
        if action in [
            "PickupObject",
            "OpenObject",
            "PutObject",
            "CloseObject",
            "ToggleObjectOn",
            "ToggleObjectOff",
        ]:
            object1 = js["api_action"]["objectId"].split("|")[0]
            object1 = object.id_to_str[object1]
        else:
            object1 = None

        if action in ["PutObject"]:
            object2 = js["api_action"]["receptacleObjectId"].split("|")[0]
            object2 = object.id_to_str[object2]
        else:
            object2 = None

        return LowLevelAction(
            action=action, images=images, object1=object1, object2=object2
        )


@dataclass
class HighLevelAction:
    human_descriptions: list[str]
    actions: list[LowLevelAction]

    action: str
    object1: str
    beg_hand_content: Optional[str]
    end_hand_content: Optional[str]
    scene: str

    beg_location: Optional[str]
    end_location: Optional[str]

    object2: Optional[str] = None

    # Only used when editing tasks by hand
    # If None, we generate a description from the action and objects
    _description: Optional[str] = None
    # Only used for substitutions
    # If None, we inherit the image path from the video we're in
    _images_path: Optional[Path] = None

    def __hash__(self) -> int:
        return hash((self.action, self.object1, self.object2))

    def __eq__(self, other) -> bool:
        return (
            self.action == other.action
            and self.object1 == other.object1
            # If the 2nd object is None, it matches everything
            and (not self.object2 or not other.object2 or self.object2 == other.object2)
        )

    @property
    def description(self):
        if self._description:
            return self._description
        if self.action == "PickupObject":
            place = f" from the {self.object2}" if self.object2 else ""
            return f"we pick up a {self.object1}" + place
        elif self.action == "ToggleObject":
            return f"we turn the {self.object1} on"
        elif self.action == "SliceObject":
            return f"we slice the {self.object1}"
        elif self.action == "GotoLocation":
            return f"we walk to the {self.object1}"
        else:
            assert self.object2 is not None
            preposition = object.preposition.get(self.object2, "on")
            if self.action == "PutObject":
                return f"we put the {self.object1} {preposition} the {self.object2}"
            elif self.action == "CoolObject":
                return f"we cool the {self.object1} by putting it {preposition} the {self.object2} for a while"
            elif self.action == "HeatObject":
                return f"we heat the {self.object1} {preposition} the {self.object2}"
            elif self.action == "CleanObject":
                return f"we clean the {self.object1} {preposition} the {self.object2} under running water"
            else:
                raise ValueError(f"Unknown action: {self.action}")

    @staticmethod
    def from_json(
        js: dict,
        descriptions: list[str],
        low_level_actions: list[LowLevelAction],
        sliced_objects: list[str],
        scene: str,
        previous_hand_content: Optional[str],
        previous_location: Optional[str],
    ) -> "HighLevelAction":
        action = js["planner_action"]["action"]
        if action in ["HeatObject", "CoolObject", "GotoLocation"]:
            object1 = js["discrete_action"]["args"][0]
        else:
            object1 = js["planner_action"]["coordinateObjectId"][0]
        if object1 not in object.id_to_str:
            raise ValueError(f"Unknown object id: {object1}")
        object1 = object.id_to_str[object1]
        # This has some false positives, but it's the best we can do
        if object1 in sliced_objects and action != "SliceObject":
            object1 = "slice of " + object1

        object2 = None
        if action in [
            "PutObject",
            "PickupObject",
            "HeatObject",
            "CoolObject",
            "CleanObject",
        ]:
            if "coordinateReceptacleObjectId" in js["planner_action"]:
                object2 = js["planner_action"]["coordinateReceptacleObjectId"][0]
                if object2 not in object.id_to_str:
                    raise ValueError(f"Unknown object id: {object2}")
                object2 = object.id_to_str[object2]

        beg_hand_content = previous_hand_content
        if action == "PickupObject":
            end_hand_content = object1
        elif action == "PutObject":
            end_hand_content = None
        else:
            end_hand_content = previous_hand_content

        if action == "GotoLocation":
            beg_location = previous_location
            end_location = object1
        elif action in ["PickupObject", "PutObject"]:
            beg_location = object2
            end_location = beg_location
        elif action in ["CoolObject"]:
            beg_location = "fridge"
            end_location = "fridge"
        elif action in ["HeatObject"]:
            beg_location = "microwave"
            end_location = "microwave"
        elif action in ["CleanObject"]:
            beg_location = "sink basin"
            end_location = "sink basin"
        else:
            beg_location = previous_location
            end_location = previous_location

        return HighLevelAction(
            human_descriptions=descriptions,
            actions=low_level_actions,
            action=action,
            object1=object1,
            object2=object2,
            beg_hand_content=beg_hand_content,
            end_hand_content=end_hand_content,
            beg_location=beg_location,
            end_location=end_location,
            scene=scene,
        )


@dataclass
class Trajectory:
    trial_id: str
    dataset: str
    type: str

    actions: list[HighLevelAction]

    _human_overall_annotations: list[str]
    _images_path: Path
    # Only used when editing trajectories by hand
    # If None, we generate a description from the high level actions
    _description: Optional[str] = None

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.with_modified_actions(self.actions[idx])
        else:
            return self.actions[idx]

    @property
    def description(self, high_level=True, block_level=True) -> str:
        if self._description is not None:
            return self._description
        if high_level and not block_level:
            actions = self.actions
            if len(actions) == 1:
                return actions[0].description.capitalize()
            last_action = actions[-1].description
            return (
                "First, "
                + ", then ".join(a.description for a in actions[:-1])
                + (", and finally " if len(actions) > 2 else ", and then ")
                + last_action
            )
        elif block_level:
            actions = [a for a in self.actions if a.action != "GotoLocation"]
            if len(actions) == 1:
                return actions[0].description.capitalize()
            last_action = actions[-1].description
            return (
                "First, "
                + ", then ".join(a.description for a in actions[:-1])
                + (", and finally " if len(actions) > 2 else ", and then ")
                + last_action
            )
        else:
            raise NotImplementedError

    @property
    def n_frames(self) -> int:
        return sum(len(lla.images) for hla in self.actions for lla in hla.actions)

    def video(self, min_frames: int = 32) -> ImageSequenceClip:
        clip_cache, frames = {}, []
        k = (min_frames // self.n_frames) + 1
        for hla in self.actions:
            path = hla._images_path or self._images_path
            clip = clip_cache.setdefault(path, VideoFileClip(str(path)))
            frame_indices = [img for lla in hla.actions for img in lla.images]
            fps = clip.fps
            frames += [f for i in frame_indices for f in [clip.get_frame(i / fps)] * k]
        return ImageSequenceClip(frames, fps=fps)

    def substitute(self, index: int, action: HighLevelAction) -> "Trajectory":
        if index > len(self):
            raise ValueError("Index out of range")
        return self.with_modified_actions(
            self.actions[:index] + [action] + self.actions[index + 1 :]
        )

    def filter_low_level(self, predicate) -> "Trajectory":
        new_actions = []
        for action in self.actions:
            new_low_level_actions = [lla for lla in action.actions if predicate(lla)]
            new_actions.append(
                HighLevelAction(
                    human_descriptions=action.human_descriptions,
                    actions=new_low_level_actions,
                    action=action.action,
                    object1=action.object1,
                    object2=action.object2,
                    beg_hand_content=action.beg_hand_content,
                    end_hand_content=action.end_hand_content,
                    beg_location=action.beg_location,
                    end_location=action.end_location,
                    scene=action.scene,
                )
            )
        return self.with_modified_actions(new_actions)

    def with_modified_actions(self, actions: list[HighLevelAction]):
        return Trajectory(
            self.trial_id, self.dataset, self.type, actions, [], self._images_path
        )

    def reverse(self, description: str):
        hlas = []
        for hla in self.actions[::-1]:
            llas = deepcopy(hla.actions[::-1])
            for lla in llas:
                lla.images = lla.images[::-1]
            hlas.append(
                HighLevelAction(
                    human_descriptions=[],
                    actions=llas,
                    action=hla.action,
                    object1=hla.object1,
                    object2=hla.object2,
                    beg_hand_content=hla.end_hand_content,
                    end_hand_content=hla.beg_hand_content,
                    beg_location=hla.end_location,
                    end_location=hla.beg_location,
                    scene=hla.scene,
                )
            )
        new_self = self.with_modified_actions(hlas)
        new_self._description = description
        return new_self

    @staticmethod
    def from_file(path: Path):
        with open(path, "r") as f:
            js = json.load(f)

            high_to_low, low_to_images = {}, {}
            for i, img in enumerate(js["images"]):
                high_to_low.setdefault(img["high_idx"], set()).add(img["low_idx"])
                low_to_images.setdefault(img["low_idx"], []).append(i)

            action_count = len(js["turk_annotations"]["anns"][0]["high_descs"])
            descriptions = [
                [desc["high_descs"][i] for desc in js["turk_annotations"]["anns"]]
                for i in range(action_count)
            ]

            # Here we assume that:
            # - the video dir is videos
            # - other than that, the video is stored in a path that mirrors `path`
            # - the video is named video.mp4
            parts = list(path.parts)
            parts[0] = "videos"
            images_path = list(Path(*parts[:-1]).glob("**/video.mp4"))[0]

            low_level_actions = [
                LowLevelAction.from_json(action, low_to_images[i])
                for i, action in enumerate(js["plan"]["low_actions"])
            ]

            high_level_actions, sliced_objects, hand_content, location = (
                [],
                [],
                None,
                None,
            )
            for action, description in zip(js["plan"]["high_pddl"], descriptions):
                lla_indices = sorted(list(high_to_low[action["high_idx"]]))
                hla = HighLevelAction.from_json(
                    action,
                    description,
                    [low_level_actions[i] for i in lla_indices],
                    sliced_objects.copy(),
                    previous_hand_content=hand_content,
                    previous_location=location,
                    scene=js["scene"]["floor_plan"],
                )
                hla._images_path = images_path
                hand_content = hla.end_hand_content
                location = hla.end_location
                high_level_actions.append(hla)
                if hla.action == "SliceObject":
                    sliced_objects.append(hla.object1)

            for prev, next in zip(high_level_actions[:-1], high_level_actions[1:]):
                if prev.end_location in object.locations:
                    next.beg_location = prev.end_location
                elif next.beg_location:
                    prev.end_location = next.beg_location
                elif prev.end_location:
                    next.beg_location = prev.end_location

            return Trajectory(
                trial_id=js["task_id"],
                dataset=parts[1],
                type=js["task_type"],
                actions=high_level_actions,
                _human_overall_annotations=[
                    a["task_desc"] for a in js["turk_annotations"]["anns"]
                ],
                _images_path=images_path,
            )


def to_action_blocks(actions: list[HighLevelAction]) -> list[list[HighLevelAction]]:
    grouped_actions, group = [], []
    for a in actions:
        if group and group[-1].action != "GotoLocation":
            grouped_actions.append(group)
            group = []
        group.append(a)
    if group:
        grouped_actions.append(group)
    return grouped_actions


def shorten_trajectories(trajectories: list[Trajectory], to: int) -> list["Trajectory"]:
    result = []
    for trajectory in trajectories:
        blocks = to_action_blocks(trajectory.actions)
        if len(blocks) < to:
            continue
        for beg in range(len(blocks) - to + 1):
            end = beg + to
            new_actions = [a for block in blocks[beg:end] for a in block]
            new_trajectory = trajectory.with_modified_actions(new_actions)
            result.append(new_trajectory)
    return result
