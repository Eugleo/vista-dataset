import os
import json
import yaml

# Within a given directory, find all the files ending in _data.json.
# Each of these contains a list of dictionaries with a "path" key.
# Verify that each of these paths actually exists.
# Warn if there are duplicate paths.

CHECK_UNUSED_FILES = True
CHECK_FILES_NOT_CORRUPTED = False
if CHECK_FILES_NOT_CORRUPTED:
    # from .....evan_gunter.compress_vids import VideoFileClip
    from moviepy.editor import VideoFileClip

def check_known_duplicate(configs, paths):
    # known duplicates:
    # - everything in sliced v whole is also in objects. paths should be in foundation/pick_object
    # - every put in pick v put is also in containers. paths should be in foundation/put_object

    if len(configs) > 2:
        return False
    c1_p, c2_p = sorted(configs)
    c1, c2 = "/".join(c1_p.split("/")[:-1]), "/".join(c2_p.split("/")[:-1])
    if c1 == "foundation/objects" and c2 == "foundation/sliced_v_whole" and all(p.startswith("/data/datasets/vlm_benchmark/videos/real_life/foundation/pick_object") for p in paths):
        return True
    if c1 == "foundation/containers" and c2 == "foundation/pick_v_put" and all(p.startswith("/data/datasets/vlm_benchmark/videos/real_life/foundation/put_object") for p in paths):
        return True


paths = {}

def validate_paths(directory):
    if not os.path.exists(directory):
        print("\033[91m" + f"Directory not found: {directory}" + "\033[0m")
        return
    bad_count = 0
    error_messages = []
    valid_paths = []
    all_data = []
    for root, _, files in os.walk(directory):
        root_without_original_part = root.replace("/data/datasets/vlm_benchmark/tasks/real_life/", "")
        if root_without_original_part.startswith("."):
            print(f"Ignoring hidden directory: {root_without_original_part}")
            continue
        for file in files:
            if file.endswith("_data.json"):
                with open(os.path.join(root, file)) as f:
                    data = json.load(f)
                    for item in data:
                        path = os.path.join('/data/datasets/vlm_benchmark/videos/real_life', item["path"])
                        if path not in paths:
                            paths[path] = []
                        paths[path].append(f"{root_without_original_part}/{file}")
                        if not os.path.exists(path):
                            msg = "\033[91m" + f"File not found: {path} (from config {root_without_original_part}/{file})" + "\033[0m"
                            error_messages.append(msg)
                            bad_count += 1
                        else:
                            valid_paths.append(path)
                            all_data.append((os.path.join(root, file), file, item))

    # warn for duplicates in yellow
    dup_dict = {}
    for p, f in paths.items():
        if len(f) == 1:
            continue
        ft = tuple(f)
        if ft not in dup_dict:
            dup_dict[ft] = []
        dup_dict[ft].append(p)

    for f, p in dup_dict.items():
        if check_known_duplicate(f, p):
            continue
        newline = '\n'
        print(f"Configs:\n{newline.join(list(f))}\nShared paths:\n\033[93m{newline.join(p)}\033[0m\n")

    if CHECK_FILES_NOT_CORRUPTED:
        print("Checking for corrupted files...")
        for path in valid_paths:
            try:
                with VideoFileClip(path) as clip:
                    pass
                print(".", end="", flush=True)
            except Exception as e:
                print("x", end="", flush=True)
                msg = "\033[91m" + f"File corrupted: {path} (from config {root_without_original_part}/{file})" + "\033[0m"
                error_messages.append(msg)
                bad_count += 1

    # see if there are any .mp4 files in the directory that aren't in the config
    if CHECK_UNUSED_FILES:
        unused_msgs = []
        for root, _, files in os.walk("/data/datasets/vlm_benchmark/videos/real_life"):
            # ignore directories which start with .
            if any(part.startswith(".") for part in root.split("/")):
                continue
            for file in files:
                if file.endswith(".mp4"):
                    path = os.path.join(root, file)
                    if path not in valid_paths:
                        msg = "\033[93m" + f"Unused file: {path}" + "\033[0m"
                        unused_msgs.append(msg)
        for msg in unused_msgs:
            print(msg)

    # print errors
    print()
    for msg in error_messages:
        print(msg)
    print()

    ok_count = sum(len(f) for f in paths.values()) - bad_count

    print("\033[92m" + f"OK: {ok_count}" + "\033[0m")
    if bad_count > 0:
        print("\033[91m" + f"Bad: {bad_count}" + "\033[0m")
    else:
        print("\033[92m" + "All paths are valid" + "\033[0m")

    datasheet_path = "/data/datasets/vlm_benchmark/tasks/real_life/datasheet.json"
    # write out the contents of all the valid json data files to a datasheet
    all_data_ = []
    for root, _, files in []: # os.walk(directory):
        # Ignore hidden directories
        if any(part.startswith(".") for part in root.split("/")):
            continue
        for file in files:
            if file.endswith("_data.json"):
                with open(os.path.join(root, file)) as f:
                    data = json.load(f)
                    all_data.extend([(os.path.join(root, file), file, d) for d in data])
    # for each entry, we want to write
    # video: <path>
    # labels: [
    #   task: <task>,
    #   label: <label>
    #   description: <description>
    # ]
    # environment: real_life
    # n_steps: <level, or 1 for foundation/extrapyramidal>
    # group: <group>

    datasheet_data = []
    for fp, fn, entry in all_data:
        path = entry["path"]
        # we will read the natural language description from the corresponding yaml file
        yaml_path = fp[:-len("_data.json")] + ".yaml"
        if not os.path.exists(yaml_path):
            print(f"Warning: yaml file not found: {yaml_path}")
            continue
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        try:
            label_desc = yaml_data["label_prompts"][entry["label"]]
        except KeyError as e:
            # include the label, list of labels, and the yaml file
            raise ValueError(f"Label '{entry['label']}' not in {list(yaml_data['label_prompts'].keys())} in {yaml_path}")
        
        group = yaml_path.split("/")[7]

        datasheet_entry = {
            "video": path,
            "labels": [
                {
                    # task is the name of the json file it's from
                    "task": fn[:-len("_data.json")],
                    "label": entry["label"],
                    "description": label_desc,
                }
            ],
            "environment": "real_life" if not "habitat" in path else "habitat",
            "n_steps": 1 if ("foundation" in fp or "extrapyramidal" in fp) else int(fp[fp.index("level"):].split("_")[1].split("/")[0]),
            "group": group,
        }
        datasheet_data.append(datasheet_entry)
    with open(datasheet_path, "w") as f:
        json.dump(datasheet_data, f, indent=2)

        

# validate_paths("/data/datasets/vlm_benchmark/tasks/real_life")
validate_paths("/data/datasets/vlm_benchmark/tasks/real_life")