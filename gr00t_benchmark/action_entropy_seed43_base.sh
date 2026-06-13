#!/bin/bash

conda activate libero
cd /home/sangfor/code/lyc/LIBERO/gr00t_benchmark

python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_16 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_object --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_16 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_16 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_16 --chunk_id_selector 0 --move_th 3 --seed 43

python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_12 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_object --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_12 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_12 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_12 --chunk_id_selector 0 --move_th 3 --seed 43

python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_8 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_object --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_8 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_8 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_8 --chunk_id_selector 0 --move_th 3 --seed 43

python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_4 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_object --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_4 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_4 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_4 --chunk_id_selector 0 --move_th 3 --seed 43

python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_2 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_object --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_2 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_2 --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8091 --out_path ~/Videos/libero_output/action_entropy_seed_43 --chunk_size_selector fixed_2 --chunk_id_selector 0 --move_th 3 --seed 43