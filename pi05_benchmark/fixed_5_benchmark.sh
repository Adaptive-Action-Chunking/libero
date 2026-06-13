#!/bin/bash

# conda activate libero
cd /home/sangfor/code/lyc/LIBERO/pi05_benchmark

# python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8000 --out_path ~/Videos/libero_output_pi05/base_seed_42 --chunk_size_selector fixed_5 --chunk_id_selector 0 --move_th 3 --seed 42
# python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_object --port 8000 --out_path ~/Videos/libero_output_pi05/base_seed_42 --chunk_size_selector fixed_5 --chunk_id_selector 0 --move_th 3 --seed 42
python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8000 --out_path ~/Videos/libero_output_pi05/base_seed_42 --chunk_size_selector fixed_5 --chunk_id_selector 0 --move_th 3 --seed 42
python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8000 --out_path ~/Videos/libero_output_pi05/base_seed_42 --chunk_size_selector fixed_5 --chunk_id_selector 0 --move_th 3 --seed 42

python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8000 --out_path ~/Videos/libero_output_pi05/base_seed_43 --chunk_size_selector fixed_5 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_object --port 8000 --out_path ~/Videos/libero_output_pi05/base_seed_43 --chunk_size_selector fixed_5 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8000 --out_path ~/Videos/libero_output_pi05/base_seed_43 --chunk_size_selector fixed_5 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8000 --out_path ~/Videos/libero_output_pi05/base_seed_43 --chunk_size_selector fixed_5 --chunk_id_selector 0 --move_th 3 --seed 43

python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8000 --out_path ~/Videos/libero_output_pi05/base_seed_44 --chunk_size_selector fixed_5 --chunk_id_selector 0 --move_th 3 --seed 44
python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_object --port 8000 --out_path ~/Videos/libero_output_pi05/base_seed_44 --chunk_size_selector fixed_5 --chunk_id_selector 0 --move_th 3 --seed 44
python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8000 --out_path ~/Videos/libero_output_pi05/base_seed_44 --chunk_size_selector fixed_5 --chunk_id_selector 0 --move_th 3 --seed 44
python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8000 --out_path ~/Videos/libero_output_pi05/base_seed_44 --chunk_size_selector fixed_5 --chunk_id_selector 0 --move_th 3 --seed 44

