import os

object_data = [
    {
        'obj': 'desk lamp',
        'a_or_an_obj': 'a desk lamp',
        'task_specific': 'near a desk',
    },
    {
        'obj': 'floor lamp',
        'a_or_an_obj': 'a floor lamp',
        'task_specific': 'near a door'
    },
    {
        'obj': 'microwave',
        'a_or_an_obj': 'a microwave',
        'task_specific': 'near a countertop',
        'task_specific_2': ' We can see this because the light inside\\n  has turned on.'
    },
    {
        'obj': 'faucet',
        'a_or_an_obj': 'a faucet',
        'task_specific': 'in a kitchen'
    }
]

task_dir = "/data/datasets/vlm_benchmark/tasks/real_life/foundation/toggle"

# for each task, we call gen_from_template.sh with the path to the task directory and the object data
for task in object_data:
    obj = task['obj']
    a_or_an_obj = task['a_or_an_obj']
    task_specific = task['task_specific']
    if 'task_specific_2' in task:
        os.system(f"/data/datasets/vlm_benchmark/tasks/real_life/gen_from_template.sh {task_dir} \"{obj}\" \"{a_or_an_obj}\" \"{task_specific}\" \"{task['task_specific_2']}\"")
    else:
        os.system(f"/data/datasets/vlm_benchmark/tasks/real_life/gen_from_template.sh {task_dir} \"{obj}\" \"{a_or_an_obj}\" \"{task_specific}\"")
