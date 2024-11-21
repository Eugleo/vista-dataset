import os
import re
import itertools
import shutil
import json

# the parsing is much easier if there aren't any objects with underscores in their names
# for f in *butter_knife*; do mv -- "$f" "${f//butter_knife/butterknife}"; done
# read in all the videos from
taskpath = "/data/datasets/vlm_benchmark/tasks/real_life"
vids_dir = "/data/datasets/vlm_benchmark/videos/real_life/extrapyramidal/object_tracking/scramble"

def get_data_files():
    vids_files = []

    # read every file in vids_dir
    for root, dirs, files in os.walk(vids_dir):
        for file in files:
            if file.endswith(".mp4"):
                if not file.startswith("scramble_"):
                    raise ValueError(f"unexpected video file: {file}")
                vids_files.append(file)
    
    return vids_files

# although we have a fixed list of tasks, we originally made the list like this
def parse_scramble(vids_files):
    # now, we parse the list of files to get the task names
    # the filenames will have the format:
    # scramble_<object>_<number of objects>_<label>_<n>.mp4
    # we want to get the object, number of objects, label, and extra index for multiples
    tasks = []
    for file in vids_files:
        match = re.match(r"scramble_(\w+)_(\d+)_(\w+)_(\d+)\.mp4", file)
        if not match:
            raise ValueError(f"nonmatching video file: {file}")
        obj = match.group(1)
        num = int(match.group(2))
        label = match.group(3)
        idx = int(match.group(4))
        tasks.append((obj, num, label, idx))

    return tasks

def decode_shorthand(task_desc):

    objects = {
        "ap": "apple",
        "butterknife": "butter knife",
        "can": "can",
        "h": "hammer",
        "mug": "mug",
        "p": "potato",
        "potsl": "slice of potato",
    }

    toggleable_objects = {
        "dl": "desk lamp",
        "fl": "floor lamp",
    }

    locations = {
        "c": "counter",
        "f": "freezer",
        "m": "microwave",
        "s": "sink",
        "t": "table",
    }

    actions = {
        "pick": "pickup",
        "put": "put",
        "heat": "heat",
        "cool": "cool",
        "clean": "clean",
        "slice": "slice",
        "tog": "toggle",
    }

    actions_with_loc = ["pickup", "put"]

    decode_dict = {**objects, **toggleable_objects, **locations, **actions}

    subtasks = []
    components = task_desc.split("_")
    while True:
        if len(components) == 0:
            break
        component = components.pop(0)
        # TODO remove
        if component == "toggle":
            print("warning: toggle")
            component = "tog"
        # we parse one verb at a time
        assert component in list(actions.keys()), f"unexpected component: {component}"
        verb = decode_dict[component]
        obj_raw = components.pop(0)
        # TODO remove:
        if obj_raw == "butter":
            print("warning: butter")
            components.pop(0)
            obj_raw = "butterknife"
        obj = decode_dict[obj_raw]

        if verb in actions_with_loc:
            loc = decode_dict[components.pop(0)]
        else:
            loc = None

        subtasks.append((verb, obj, loc))

    return subtasks

crystal_path = f"{taskpath}/_scramble_vid_paths.txt"

def crystallize_videos():
    vid_paths = get_data_files()
    # dump the video names to a file
    with open(crystal_path, "w") as f:
        f.write("\n".join(vid_paths))

def load_crystallized_videos():
    with open(crystal_path, "r") as f:
        vid_paths = [vp.strip() for vp in f.readlines()]
    # parse the video names
    vids_parsed = parse_scramble(vid_paths)
    vp = zip(vids_parsed, vid_paths)
    out_vp = sorted(vp, key=lambda x: x[0])
    out_vids_parsed, out_vid_paths = zip(*out_vp)
    return out_vids_parsed, out_vid_paths

def get_task_label(subtasks):
    locs_in = ["freezer", "microwave", "sink"]
    locs_on = ["counter", "table"]

    descs = []
    for verb, obj, loc in subtasks:
        if loc is not None:
            if verb == "pickup":
                # if obj starts with a vowel, use "an" instead of "a"
                article = "an" if obj[0] in "aeiou" else "a"
                subtask_desc = f"we pick up {article} {obj} from the {loc}"
            elif verb == "put":
                preposition = "in" if loc in locs_in else "on"
                subtask_desc = f"we put the {obj} {preposition} the {loc}"
            else:
                raise ValueError(f"unexpected verb: {verb}")
        else:
            subtask_desc = f"we {verb} the {obj}"
        descs.append(subtask_desc)

    # The description is formatted:
    # First, [subtask], then [subtask], then [subtask]..., and finally [subtask].
    # except when there are only two subtasks, in which case it is:
    # First, [subtask], and then [subtask].
    subtask_descs = []
    if len(descs) == 2:
        d1, d2 = descs
        subtask_descs.append(f"First, {d1}")
        subtask_descs.append(f", and then {d2}")
    else:
        for i, desc in enumerate(descs):
            if i == 0:
                subtask_descs.append(f"First, {desc}")
            elif i == len(descs) - 1:
                subtask_descs.append(f", and finally, {desc}")
            else:
                subtask_descs.append(f", then {desc}")
        
    task_desc = "".join(subtask_descs)
       
    # wrap at a space before 80 characters, and indent to 4 spaces
    wrapped_desc = re.sub(r"(.{1,80})(?:\s|$)", r"\1\n    ", task_desc)
    return wrapped_desc

def get_label_block(task_group):
    print(f"task group: {task_group}")
    # labels = [get_task_label(task) for task, path in task_group]
    labels = [task for task, path in task_group]
    idxs = [label for (obj, num, label, idx) in labels]
    label_block = format_labels(labels, idxs)
    return label_block, list(zip(idxs, [path for task, path in task_group]))

def format_labels(labels, idxs):
    out = "  "
    for idx, desc in zip(idxs, labels):
        out += f"{idx}: {desc}"
        out = out[:-2]  # dumb hack to remove extra indentation
    return out

def make_one_config(obj, num_objects, task_group):
    template_path = f"{taskpath}/extrapyramidal/object_tracking/_scramble_template.yaml"

    task_name = f"scramble_{num_objects}_{obj}"

    yaml_config_path = f"{taskpath}/extrapyramidal/object_tracking/scramble/{task_name}.yaml"
    json_config_path = f"{taskpath}/extrapyramidal/object_tracking/scramble/{task_name}_data.json"

    labels, paths = get_label_block(task_group)

    # write the yaml config
    shutil.copyfile(template_path, yaml_config_path)
    # replace "$(OBJ)" in copy of template with the list of labels
    # sed -i "s/\$(OBJ)/$replacement_word/g" "$input_dir/$template" > "$output_file"
    # but labels contains newlines, so we need to do it in python
    with open(yaml_config_path, "r") as f:
        template = f.read()
    with open(yaml_config_path, "w") as f:
        f.write(template.replace("$(OBJ)", labels))

    path_dict = [{"path": f"extrapyramidal/object_tracking/scramble/{path}", "label": idx} for idx, path in paths]
    with open(json_config_path, "w") as f:
        json.dump(path_dict, f, indent=4)


crystallize_videos()
vids_parsed, vid_paths = load_crystallized_videos()

print(f"vids_parsed: {vids_parsed}")
print(f"vid_paths: {vid_paths}")

tasks = list(zip(vids_parsed, vid_paths))

# now group the tasks by object and number of objects
grouped_tasks = [(key, [task for task in group]) for key, group in itertools.groupby(tasks, key=lambda x: (x[0][0], x[0][1]))]

print("grouped_tasks")
for key, group in grouped_tasks:
    print(key)
    for task in group:
        print(task)
    print()

task_list = []
for (obj, num), tasks in grouped_tasks:
    make_one_config(obj, num, tasks)
    task_list.append((obj, num, f"scramble_{num}_{obj}"))

print("tasks:")
for lvl, gn, task in task_list:
    print(f"- extrapyramidal/object_tracking/scramble/{task}")

# print("grouped tasks:")
# for key, group in grouped_tasks:
#     print(key)
#    for task in group:
#         print(task)
#     print()
