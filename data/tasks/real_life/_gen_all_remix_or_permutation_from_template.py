import os
import re
import itertools
import shutil
import json
import sys

from _utils import decode_shorthand

# the parsing is much easier if there aren't any objects with underscores in their names
# for f in *butter_knife*; do mv -- "$f" "${f//butter_knife/butterknife}"; done

taskpath = "/data/datasets/vlm_benchmark/tasks/real_life"

# determine whether 'remix' or 'permutation' was input
if len(sys.argv) != 2 or sys.argv[1] not in ["remix", "permutation"]:
    raise ValueError("expected one argument, 'remix' or 'permutation'")
task_type = sys.argv[1]

if task_type == "remix":
    vids_dir = "/data/datasets/vlm_benchmark/real_life/remixes"
    prefix = "remix_"
    regex = r"remix_l(\d+)_(\d+)_(.+)\.mp4"
    crystal_path = f"{taskpath}/_remix_vid_paths.txt"
elif task_type == "permutation":
    vids_dir = "/data/datasets/vlm_benchmark/real_life/permutations"
    prefix = "perm_"
    regex = r"perm_l(\d+)_(\d+)_(.+)\.mp4"
    crystal_path = f"{taskpath}/_perm_vid_paths.txt"
else:
    assert False, "unreachable"

def get_data_files():
    vids_files = []

    # read every file in vids_dir
    for root, dirs, files in os.walk(vids_dir):
        for file in files:
            if file.endswith(".mp4"):
                if not file.startswith(prefix):
                    raise ValueError(f"unexpected video file: {file}")
                vids_files.append(file)
    
    return vids_files

# although we have a fixed list of tasks, we originally made the list like this
def parse_levels(vids_files):
    # now, we parse the list of files to get the task names
    # the filenames will have the format:
    # perm_l<level>_<group number>_<description_of_task_components_separated_by_underscores>.mp4
    # we want to get the level, group number, and description (which is the entire rest of the filename)
    tasks = []
    for file in vids_files:
        match = re.match(regex, file)
        if not match:
            raise ValueError(f"nonmatching video file: {file}")
        level = int(match.group(1))
        group = int(match.group(2))
        description = match.group(3)
        tasks.append((level, group, description))

    return tasks

def crystallize_videos():
    vid_paths = get_data_files()
    # dump the video names to a file
    with open(crystal_path, "w") as f:
        f.write("\n".join(vid_paths))

def load_crystallized_videos():
    with open(crystal_path, "r") as f:
        vid_paths = [vp.strip() for vp in f.readlines()]
    # parse the video names
    vids_parsed = parse_levels(vid_paths)
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
    labels = [get_task_label(task) for _, _, task, _ in task_group]
    idxs = [f"label_{i}" for i in range(len(labels))]
    label_block = format_labels(labels, idxs)
    return label_block, list(zip(idxs, [path for _, _, _, path in task_group]))

def format_labels(labels, idxs):
    out = "  "
    for idx, desc in zip(idxs, labels):
        out += f"{idx}: {desc}"
        out = out[:-2]  # dumb hack to remove extra indentation
    return out

def make_one_config(level, group_num, task_group):
    template_path = f"{taskpath}/_permutation_and_remix_template.yaml"

    task_name = f"{task_type}_level_{level}_group_{group_num}"

    yaml_config_path = f"{taskpath}/level_{level}/{task_type}/{task_name}.yaml"
    json_config_path = f"{taskpath}/level_{level}/{task_type}/{task_name}_data.json"

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

    path_dict = [{"path": f"{task_type}{'e' if task_type == 'remix' else ''}s/{path}", "label": idx} for idx, path in paths]
    with open(json_config_path, "w") as f:
        json.dump(path_dict, f, indent=4)


crystallize_videos()
vids_parsed, vid_paths = load_crystallized_videos()
tasks = []
for (level, group, desc), path in zip(vids_parsed, vid_paths):
    try:
        decode_shorthand(desc)
    except AssertionError as e:
        # reraise this error but indicating which level and group it came from
        raise AssertionError(f"level {level}, group {group}: {str(e)}")
    tasks.append((level, group, decode_shorthand(desc), path))

# now group the tasks by level and group
grouped_tasks = [(key, [task for task in group]) for key, group in itertools.groupby(tasks, key=lambda x: (x[0], x[1]))]

task_list = []
for (level, group_num), tasks in grouped_tasks:
    make_one_config(level, group_num, tasks)
    task_list.append((level, group_num, f"{task_type}_level_{level}_group_{group_num}"))

print("tasks:")
for lvl, gn, task in task_list:
    print(f"- level_{lvl}/{task_type}/{task}")

# print("grouped tasks:")
# for key, group in grouped_tasks:
#     print(key)
#    for task in group:
#         print(task)
#     print()
