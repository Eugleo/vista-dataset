import os

object_data = [
    {
        'obj': 'desk lamp',
        'a_or_an_obj': 'a desk lamp',
        'task_specific': 'There is no light coming from it.'
    },
    {
        'obj': 'floor lamp',
        'a_or_an_obj': 'a floor lamp',
        'task_specific': 'There is no light coming from it.'
    },
    {
        'obj': 'microwave',
        'a_or_an_obj': 'a microwave',
        'task_specific': 'Its light is not on, and the number\\n  is the same as in the previous frame, suggesting that it is not counting down.'
    },
    {
        'obj': 'faucet',
        'a_or_an_obj': 'a faucet',
        'task_specific': 'There is no water coming from it.'
    }
]

task_dir = "/data/datasets/vlm_benchmark/tasks/real_life/foundation/on_v_off"

# for each task, we call gen_from_template.sh with the path to the task directory and the object data
for task in object_data:
    obj = task['obj']
    a_or_an_obj = task['a_or_an_obj']
    task_specific = task['task_specific']
    os.system(f"/data/datasets/vlm_benchmark/tasks/real_life/gen_from_template.sh {task_dir} \"{obj}\" \"{a_or_an_obj}\" \"{task_specific}\"")
