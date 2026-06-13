#!/bin/bash

conda activate libero
cd /home/sangfor/code/lyc/LIBERO/gr00t_benchmark
python inference_client_gr00t_baseline.py --num_trials_per_task 50 --task_suite_name libero_object --port 8089 --out_path ~/Videos/libero_output/baseline
python inference_client_gr00t_baseline.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8089 --out_path ~/Videos/libero_output/baseline
python inference_client_gr00t_baseline.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8089 --out_path ~/Videos/libero_output/baseline
python inference_client_gr00t_baseline.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8089 --out_path ~/Videos/libero_output/baseline