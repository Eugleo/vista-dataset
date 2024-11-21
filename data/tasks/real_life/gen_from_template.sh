#!/bin/bash

# if there are fewer than 3 input arguments, print usage and exit
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_dir> <replacement_word> <a or an replacement_word> [task-specific replacements (optional)] [task-specific replacements 2 (optional)]"
    exit 1
fi

# Assign input arguments to variables
input_dir="$1"
replacement_word="$2"
a_or_an_replacement_word="$3"
task_specific="$4"
task_specific_2="$5"
task_specific_3="$6"

template="_template.yaml"

# Check if input file exists
if [ ! -f "$input_dir/$template" ]; then
    echo "Error: '$input_dir/$template' does not exist."
    exit 1
fi

# Create output file name
replacement_filename=$(echo "$replacement_word" | tr ' ' '_')
output_file="$input_dir/${replacement_filename}.yaml"

# Replace $(OBJ) with replacement word and write to output file
sed "s/\$(OBJ)/$replacement_word/g" "$input_dir/$template" > "$output_file"

# Replace $(A_OR_AN_OBJ) with a or an replacement word
sed -i "s/\$(A_OR_AN_OBJ)/$a_or_an_replacement_word/g" "$output_file"

# If there are task-specific replacements, replace them
sed -i "s/\$(TASK_SPECIFIC)/$task_specific/g" "$output_file"

# If there are task-specific replacements 2, replace them
sed -i "s/\$(TASK_SPECIFIC_2)/$task_specific_2/g" "$output_file"

# If there are task-specific replacements 3, replace them
sed -i "s/\$(TASK_SPECIFIC_3)/$task_specific_3/g" "$output_file"

echo "Replacement completed. Modified content written to $output_file."
