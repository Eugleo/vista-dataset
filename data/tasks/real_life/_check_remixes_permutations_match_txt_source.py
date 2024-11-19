import os
import re
import sys

from _utils import decode_shorthand

# read in the mode from the command line
if len(sys.argv) != 2:
    raise ValueError("Usage: python3 _check_remixes_permutations_match_txt_source.py <mode>")
MODE = sys.argv[1]

if MODE == "remix":
    spec_source_path = "/data/datasets/vlm_benchmark/tasks/real_life/_remixes.txt"
    files_source_path = "/data/datasets/vlm_benchmark/real_life/remixes"
    stub, stubs = "remix", "remixes"
elif MODE == "perm":
    spec_source_path = "/data/datasets/vlm_benchmark/tasks/real_life/_permutations.txt"
    files_source_path = "/data/datasets/vlm_benchmark/real_life/permutations"
    stub, stubs = "perm", "permutations"
else:
    raise ValueError(f"Unknown mode: {MODE}")

# the file is split up into chunks starting with a number and colon on
# a line alone (e.g. "2:")
# then below that it will have some blocks that are also numbered at the
# start, e.g. "remixes set 1:" or "permutations set 5:"
# and below that there will be a list of task sequences separated by newlines
# each of the tasks sequences will have a comma-separated list of tasks
# of the format verb(object, location), where location may be None.
# we want to parse each task sequence into tuples of (verb, object, location)

def parse_spec(spec_source_path):
    # read the source file
    with open(spec_source_path) as f:
        source = f.readlines()

    # first, split the file into levels
    levels = []
    level = []
    for line in source:
        if re.match(r"^\d+:", line):
            if level:
                levels.append(level)
            level = []
        level.append(line)
    levels.append(level)

    # now, parse each level
    parsed = []
    for level in levels:
        # the first line of the level contains the level number, e.g. "3:"; extract the integer
        level_number = int(level[0].strip()[:-1])

        # now, split the level into chunks. each chunk starts with "<stubs> set <chunk_id>:"
        chunks = []
        chunk = []
        for line in level[1:]:
            if re.match(rf"{stubs} set \d+:", line):
                if chunk:
                    chunks.append(chunk)
                chunk = []
            chunk.append(line)
        chunks.append(chunk)

        # now, parse each chunk
        for chunk in chunks:
            # the first line of the chunk contains the chunk id e.g. "remixes set 1:"; extract the integer
            chunk_re = rf"{stubs} set (\d+):"
            chunk_id = int(re.match(chunk_re, chunk[0]).group(1))
            # the rest of the lines are the task sequences
            for line in chunk[1:]:
                if not line.strip():  # ignore newlines
                    continue
                # we split off words from the start of line piece by piece
                task_sequence = []
                while True:
                    # start out getting the verb, which is everything up to the first open paren (except any leading spaces)
                    open_paren_index = line.index("(")
                    verb = line[:open_paren_index].strip()
                    line = line[open_paren_index+1:].strip()
                    # now get the object and location from inside the parens
                    close_paren_index = line.index(")")
                    paren_contents = line[:close_paren_index]
                    line = line[close_paren_index+1:].strip()
                    obj, loc = [p.strip() for p in paren_contents.split(",")]
                    if loc == "None":
                        loc = None
                    task_sequence.append((verb, obj, loc))
                    # if the line is empty, we're done
                    if not line:
                        break
                    # otherwise, we expect a comma and a space
                    line = line[2:]
                parsed.append((level_number, chunk_id, tuple(task_sequence)))
    return parsed

def parse_filenames(videos_source_path, stub):
    # parse the names of the video files at the source path
    # they have the format "<stub>_l<level_number>_<chunk_number>_description_of_task_sequence.mp4"
    all_video_files = os.listdir(videos_source_path)
    cond = lambda f: f.endswith(".mp4") and f.startswith(stub)
    video_files = [f for f in all_video_files if cond(f)]
    other_files = [f for f in all_video_files if not cond(f)]
    if other_files:
        # print in red
        print(f"\033[91mFound non-spec files: {other_files}\033[0m")

    # parse the level, id numbers, and task sequences (using decode_shorthand) from the filenames
    parsed = []
    for video_file in video_files:
        lvl_chunk_desc = re.match(rf"{stub}_l(\d+)_(\d+)_([^\.]+)\.mp4", video_file)
        level_number, chunk_id, description = int(lvl_chunk_desc.group(1)), int(lvl_chunk_desc.group(2)), lvl_chunk_desc.group(3)
        try:
            task_sequence = decode_shorthand(description)
        except (AssertionError, KeyError) as e:
            # add in the filename for context
            raise AssertionError(f"{e} (from filename {video_file})")
        # replace any spaces with underscores; replace "butter_knife" with "butterknife"; replace "slice_of_potato" with "potato_slice"
        task_sequence = [(verb,
                          obj
                          .replace(" ", "_")
                          .replace("butter_knife", "butterknife")
                          .replace("slice_of_potato", "potato_slice"),
                          loc) for verb, obj, loc in task_sequence]
        parsed.append((level_number, chunk_id, tuple(task_sequence)))
    return parsed

def to_dict(parsed):
    dct = {}
    for lvl, ch, desc in parsed:
        if (lvl, ch) in dct:
            dct[(lvl, ch)].append(desc)
        else:
            dct[(lvl, ch)] = [desc]
    return dct

def main():
    print(f"{stub.upper()}")
    parsed_spec = parse_spec(spec_source_path)
    parsed_files = parse_filenames(files_source_path, stub)

    # find all the elements of the spec that aren't in the files and vice versa
    spec_dict = to_dict(parsed_spec)
    files_dict = to_dict(parsed_files)
    assert sum(len(d) for _, d in spec_dict.items()) == len(parsed_spec), f"duplicate elements in spec: {len(spec_dict)} vs {len(parsed_spec)}"
    assert sum(len(d) for _, d in files_dict.items()) == len(parsed_files), f"duplicate elements in files: {len(files_dict)} vs {len(parsed_files)}"

    # for each chunk, print the number of shared elements, and the elements that differ
    groups_should_have = set(spec_dict.keys())
    # print any levels that the files have that the spec doesn't
    groups_have = set(files_dict.keys())
    groups_not_spec = sorted(list(groups_have - groups_should_have))
    groups_not_files = sorted(list(groups_should_have - groups_have))
    shared_groups = sorted(list(groups_have & groups_should_have))
    if groups_not_spec:
        print(f"Groups not in spec: {groups_not_spec}")
    if groups_not_files:
        print(f"Groups not in files: {groups_not_files}")

    for gr in shared_groups:
        descs_s = spec_dict[gr]
        descs_f = files_dict[gr]
        descs_s_set = set(descs_s)
        descs_f_set = set(descs_f)
        shared = descs_s_set & descs_f_set
        s_n_f = sorted(list(descs_s_set - descs_f_set))
        f_n_s = sorted(list(descs_f_set - descs_s_set))
        if len(s_n_f) + len(f_n_s) > 0:
            print(f"Group {gr}:")
            print(f"\033[92mShared: {len(shared)}\033[0m")
            if len(s_n_f) != len(f_n_s):
                # in this case we don't have a guess on how to line them up, so just print them
                print(f"{len(s_n_f)} in spec not files; {len(f_n_s)} in files not spec")
                newline = "\n"
                if s_n_f:
                    print(f"In spec not files:\n{newline.join(str(d) for d in s_n_f)}")
                if f_n_s:
                    print(f"In files not spec:\n{newline.join(str(d) for d in f_n_s)}")
            else:
                print(f"{len(s_n_f)} differing")
                # zip them together and print them in two columns
                max_desc_len = max(len(str(d)) for d in s_n_f)
                print(f"In spec not files: {' '*(max_desc_len-len('In spec not files:'))}  In files not spec:")
                for dns, dnf in zip(s_n_f, f_n_s):
                    print(f"{dns}{' '*(max_desc_len-len(str(dns)))}  {dnf}")  
            print()

if __name__ == "__main__":
    main()
