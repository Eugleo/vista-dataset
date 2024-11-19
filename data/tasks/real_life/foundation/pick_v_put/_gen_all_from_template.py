import itertools
import os
import subprocess

objects = [('apple', 'an apple'), ('butter knife', 'a butter knife'), ('can', 'a can'), ('hammer', 'a hammer'), ('mug', 'a mug')]

locations = [('sink', 'a sink', 'in the sink'), ('microwave', 'an open microwave', 'in the microwave'), ('freezer', 'an open freezer', 'in the freezer'), ('countertop', 'a countertop', 'on the countertop')]

task_dir = "/data/datasets/vlm_benchmark/tasks/real_life/foundation/pick_v_put"

# for each task, we call gen_from_template.sh with the path to the task directory and the object data
for (obj_name, obj_article), (location, location_article, in_location) in itertools.product(objects, locations):
    # we can't use os.system because we need to wait for the command to finish
    subprocess.run(["/data/datasets/vlm_benchmark/tasks/real_life/gen_from_template.sh", task_dir, obj_name, obj_article, location, location_article, in_location], check=True)

    # then rename the created file to include the location
    obj_label = obj_name.replace(' ', '_')
    os.rename(f"{task_dir}/{obj_label}.yaml", f"{task_dir}/{obj_label}_{location}.yaml")

# then there's also fork with only the countertop as a location
subprocess.run(["/data/datasets/vlm_benchmark/tasks/real_life/gen_from_template.sh", task_dir, "fork", "a fork", "countertop", "a countertop", "on the countertop"], check=True)
os.rename(f"{task_dir}/fork.yaml", f"{task_dir}/fork_countertop.yaml")
