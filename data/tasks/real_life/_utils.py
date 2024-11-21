import itertools
import random

objects = ['apple', 'butterknife', 'can', 'hammer', 'mug', 'potato']
toggleable_objects = ['desk_lamp', 'floor_lamp']
locations = ['table', 'counter', 'freezer', 'sink', 'microwave']
actions = ['pickup', 'put', 'heat', 'cool', 'clean', 'slice', 'toggle']

# all actions: pickup(object, location), put(object, location), heat(object), cool(object), clean(object), slice(object), toggle(object), goto(location)

def check_valid_pair(object, action):
    if action == 'heat':
        return object in ['apple', 'mug', 'potato']
    if action == 'slice':
        return object in ['apple', 'potato']
    return True


def get_task():
    """Get a single random task"""

    # start by choosing an action at random
    action = random.choice(actions)

    # now choose a random valid object and (if applicable) location
    if action == 'pickup':
        object = random.choice(objects)
        location = random.choice(locations)
    elif action == 'put':
        object = None
        location = random.choice(locations)
    elif action == 'toggle':
        object = random.choice(toggleable_objects)
        location = None
    else:
        location = None
        object = random.choice(objects)
        while not check_valid_pair(object, action):
            object = random.choice(objects)
    
    return (action, object, location)


def check_consistent(task_sequence):
    # reject if there is the same task twice in a row
    for i in range(len(task_sequence) - 1):
        if task_sequence[i] == task_sequence[i + 1]:
            return False

    # there cannot be two consecutive pickup actions without an intervening put action
    # we can only act upon an object if we have picked it up--UNLESS there is no preceding pickup action at all, in which case we assume we start out holding the object

    held_object = 'unknown'  # at the start, we don't know if we're holding an object
    for action, object, location in task_sequence:
        # print(f"action: {action}, object: {object}, held_object: {held_object}")
        if action == 'pickup':
            if held_object not in ['unknown', 'none']:
                return False
            held_object = object
        elif action == 'put': 
            if object is not None and object != held_object:
                return False
            if held_object == 'none':
                return False
            held_object = 'none'
        elif action == 'toggle':
            pass
        else:
            if held_object == 'unknown':
                held_object = object
            elif held_object != object:
                return False
    return True

def rand_iter(lst):
    while True:
        yield random.choice(lst)

def build_task_sequence(n_tasks, tasks, prefix=[]):
    task_sequence = [t for t in prefix]  # don't modify the original prefix object
    # build up a sequence of tasks one task at a time by checking for consistency
    scrambled_tasks = random.sample(tasks, len(tasks))
    # print(f"task sequence: {task_sequence}")
    for task in scrambled_tasks:
        if check_consistent(task_sequence + [task]):
            # print(f'consistent task sequence: {task_sequence + [task]}')
            task_sequence.append(task)
        else:
            # print(f"inconsistent task sequence: {task_sequence + [task]}")
            pass
        if len(task_sequence) == n_tasks:
            return task_sequence
    return None

def get_random_task_sequence(n_tasks):
    while True:
        tasks = [get_task() for _ in range(n_tasks)]
        task_sequence = build_task_sequence(n_tasks, tasks)
        if task_sequence:
            return task_sequence
        
def format_task_sequence_set(task_sequence_set):
    return [", ".join(f"{action}({object}, {location})" for action, object, location in task_sequence) for task_sequence in task_sequence_set]
        
def limit_slices_sets(sets_maker):
    SLICE_LIMIT = 3
    # make sure i don't have to slice more than SLICE_LIMIT objects, i.e. there are no more than SLICE_LIMIT slice actions in all of the permutations/remixes
    while True:
        sets = sets_maker()
        num_slices = sum(sum(sum(action == 'slice' for action, _, _ in permutation) for permutation in set) for set in sets)
        if num_slices <= SLICE_LIMIT:
            return sets
        else:
            pass
            # print(f"too many slices: {num_slices}")

def concretize_puts(task_sequence):
    task_sequence_put_concretized = []
    held_object = 'unknown'
    for action, object, location in task_sequence:
        if action == 'put':
            object = held_object
        elif action == 'pickup' or object != None:
            held_object = object
        task_sequence_put_concretized.append((action, object, location))
    return task_sequence_put_concretized

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
            print(f"warning: toggle")
            component = "tog"
        # we parse one verb at a time
        assert component in list(actions.keys()), f"unexpected component in {task_desc}: {component}"
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
