import itertools
import random
import math

from _utils import get_task, limit_slices_sets, build_task_sequence, format_task_sequence_set, concretize_puts

# candidate_sequence = build_task_sequence(3, (get_task() for _ in itertools.count()))
# print(f"task sequence: {candidate_sequence}")

def get_permutations(n_tasks, n_permutations):
    TIMEOUT = min(1000, 3 * math.factorial(n_tasks))
    print(f"timeout: {TIMEOUT}")
    while True:
        # could use get_random_task_sequence for this probably
        # get a single list of tasks
        tasks = [get_task() for _ in range(n_tasks)]

        # get one legal permutation. we will use this to fix the object in each put action
        for _ in range(100):  # also a timeout
            task_sequence = build_task_sequence(n_tasks, tasks)
            if task_sequence:
                break
        else:
            # print("timed out on single sequence")
            continue
        
        task_sequence_put_concretized = concretize_puts(task_sequence)

        print(f"tasks: {task_sequence_put_concretized}")

        # now, see if we can find n_permutations distinct consistent permutations
        permutations = [task_sequence_put_concretized]
        for _ in range(TIMEOUT*n_permutations):
            # print(f"tasks: {tasks}")
            task_sequence = build_task_sequence(n_tasks, task_sequence_put_concretized)
            if task_sequence is not None and task_sequence not in permutations:
                permutations.append(task_sequence)

            if len(permutations) == n_permutations:
                # check that not all permutations start or end with the same action
                start_actions = [permutation[0][0] for permutation in permutations]
                end_actions = [permutation[-1][0] for permutation in permutations]
                if len(set(start_actions)) > 1 and len(set(end_actions)) > 1:
                    return permutations
                else:
                    print("all permutations start or end with the same action")
                    break
        # if we timed out, try again
        print("timed out")

def get_n_permutation_sets(n_sets, n_tasks, n_permutations):
    permutations_set = []
    while len(permutations_set) < n_sets:
        permutations = get_permutations(n_tasks, n_permutations)
        # make sure that there is no permutation with all the same actions (i.e. when sorted is the same as the new one)
        if any(sorted(permutations[0]) == sorted(ps[0]) for ps in permutations_set):
            print("duplicate set")
        else:
            permutations_set.append(permutations)
    return permutations_set

newline = '\n'  # because f-strings don't support backslashes
# permutations = get_permutations(5, 3)
# print(f"permutations:\n{newline.join(format_task_sequence_set(permutations))}\n")

permutations_set = limit_slices_sets(lambda: get_n_permutation_sets(8, 4, 3))
for i, permutations in enumerate(permutations_set):
    print(f"permutations set {i + 1}:\n{newline.join(format_task_sequence_set(permutations))}\n")
