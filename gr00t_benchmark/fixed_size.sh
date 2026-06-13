#!/bin/bash

# conda activate libero
cd /home/sangfor/code/lyc/LIBERO/gr00t_benchmark
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_object --port 8089 --out_path ~/Videos/libero_output/fixed_size_baseline_0928/ --chunk_size_selector fixed_16 --chunk_id_selector 0
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8089 --out_path ~/Videos/libero_output/fixed_size_baseline_0928/ --chunk_size_selector fixed_16 --chunk_id_selector 0
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8089 --out_path ~/Videos/libero_output/fixed_size_baseline_0928/ --chunk_size_selector fixed_16 --chunk_id_selector 0
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8089 --out_path ~/Videos/libero_output/fixed_size_baseline_0928/ --chunk_size_selector fixed_16 --chunk_id_selector 0

python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_object --port 8089 --out_path ~/Videos/libero_output/fixed_size_baseline_0928/ --chunk_size_selector fixed_12 --chunk_id_selector 0
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8089 --out_path ~/Videos/libero_output/fixed_size_baseline_0928/ --chunk_size_selector fixed_12 --chunk_id_selector 0
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8089 --out_path ~/Videos/libero_output/fixed_size_baseline_0928/ --chunk_size_selector fixed_12 --chunk_id_selector 0
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8089 --out_path ~/Videos/libero_output/fixed_size_baseline_0928/ --chunk_size_selector fixed_12 --chunk_id_selector 0

python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_object --port 8089 --out_path ~/Videos/libero_output/fixed_size_baseline_0928/ --chunk_size_selector fixed_8 --chunk_id_selector 0
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8089 --out_path ~/Videos/libero_output/fixed_size_baseline_0928/ --chunk_size_selector fixed_8 --chunk_id_selector 0
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8089 --out_path ~/Videos/libero_output/fixed_size_baseline_0928/ --chunk_size_selector fixed_8 --chunk_id_selector 0
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8089 --out_path ~/Videos/libero_output/fixed_size_baseline_0928/ --chunk_size_selector fixed_8 --chunk_id_selector 0
