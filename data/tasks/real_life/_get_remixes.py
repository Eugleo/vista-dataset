import math

from _utils import build_task_sequence, get_random_task_sequence, get_task, format_task_sequence_set, limit_slices_sets, concretize_puts

# (number of prefix-0 remixes, number of prefix-i remixes for every i > 0)
level_to_num_videos = {
    2: (4, 4),
    3: (2, 3),
    4: (2, 2),
    5: (4, 1),
    6: (3, 1),
    7: (2, 1),
    8: (1, 1),
}


def get_remixes(n_tasks):
    TIMEOUT = min(1000, 3 * math.factorial(n_tasks))
    print(f"timeout: {TIMEOUT}")
    while True:
        # get a single legal task sequence
        task_sequence = None
        while task_sequence is None or len(set(task_sequence)) != len(task_sequence):
            task_sequence = get_random_task_sequence(n_tasks)

        core_task_sequence = concretize_puts(task_sequence)
        print(f"core task sequence: {core_task_sequence}")

        remixes = [core_task_sequence]

        prefix_0_count, prefix_i_count = level_to_num_videos[n_tasks]

        n_remixes = 1 + prefix_0_count + prefix_i_count * (n_tasks - 1)

        # now get all the prefix-0 remixes, i.e. more random task sequences
        while len(remixes) < prefix_0_count + 1:
            task_sequence = get_random_task_sequence(n_tasks)
            if task_sequence not in remixes and task_sequence[0] != core_task_sequence[0]:
                remixes.append(task_sequence)
                print(f"got prefix-0 remix {task_sequence}")

        # now get all the prefix-i remixes, i.e. more task sequences with the same prefix
        for i in range(1, n_tasks):
            this_level_remix_count = 0
            timeout_count = 0
            while timeout_count < TIMEOUT:
                print(f"timeout count: {timeout_count}")
                # the 1000 here is just a hack; really we want all the tasks, but this is close enough
                task_sequence = None
                # repeated tasks are not allowed
                while (task_sequence is None or len(set(task_sequence)) != len(task_sequence)) and timeout_count < 100:
                    task_sequence = build_task_sequence(n_tasks, [get_task() for _ in range(1000)], prefix=core_task_sequence[:i])
                    timeout_count += 1
                else:
                    if timeout_count == 100:
                        print("timed out mini--STARTING OVER")
                        return get_remixes(n_tasks)
               
                if task_sequence is not None and task_sequence not in remixes and task_sequence[i] != core_task_sequence[i]:
                    remixes.append(task_sequence)
                    this_level_remix_count += 1
                    print(f"got prefix-{i} remix {task_sequence}")
                else:
                    if task_sequence is None:
                        # print("none")
                        pass
                    elif task_sequence in remixes:
                        print(f"rejected: repeated task sequence {task_sequence}")
                    elif task_sequence[i] == core_task_sequence[i]:
                        print(f"rejected: prefix longer than {i} in {task_sequence}")
                    elif len(set(task_sequence)) != len(task_sequence):
                        print(f"rejected: repeated task in {task_sequence}")
                    else:
                        assert False, "unreachable"
                    timeout_count += 1

                if this_level_remix_count == prefix_i_count:
                    break
            else:
                # start over--probably we're forced into repeating tasks for consistency
                print("STARTING OVER")
                return get_remixes(n_tasks)

        assert len(remixes) == n_remixes
        return remixes
    
def get_n_remix_sets(n_sets, n_tasks):
    remixes_set = []
    while len(remixes_set) < n_sets:
        remixes = get_remixes(n_tasks)
        # make sure that there are no duplicate core sequences
        if any(remixes[0] == rs[0] for rs in remixes_set):
            print("duplicate set")
        else:
            remixes_set.append(remixes)

    return remixes_set

newline = "\n"
remixes_set = limit_slices_sets(lambda: get_n_remix_sets(3, 4))
for i, remixes in enumerate(remixes_set):
    print(f"remixes set {i + 1}:\n{newline.join(format_task_sequence_set(remixes))}\n")