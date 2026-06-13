#!/bin/bash

# conda activate libero
cd /home/sangfor/code/lyc/LIBERO/pi05_benchmark

# python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8000 --out_path ~/Videos/libero_output_pi05/action_entropy_mv4_seed_42 --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 4 --seed 42
# python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_object --port 8000 --out_path ~/Videos/libero_output_pi05/action_entropy_mv4_seed_42 --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 4 --seed 42
# python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8000 --out_path ~/Videos/libero_output_pi05/action_entropy_mv4_seed_42 --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 4 --seed 42
# python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8000 --out_path ~/Videos/libero_output_pi05/action_entropy_mv4_seed_42 --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 4 --seed 42

# python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8000 --out_path ~/Videos/libero_output_pi05/action_entropy_mv4_seed_43 --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 4 --seed 43
# python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_object --port 8000 --out_path ~/Videos/libero_output_pi05/action_entropy_mv4_seed_43 --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 4 --seed 43
# python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8000 --out_path ~/Videos/libero_output_pi05/action_entropy_mv4_seed_43 --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 4 --seed 43
python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8000 --out_path ~/Videos/libero_output_pi05/action_entropy_mv4_seed_43 --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 4 --seed 43

# python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8000 --out_path ~/Videos/libero_output_pi05/action_entropy_mv4_seed_44 --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 4 --seed 44
# python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_object --port 8000 --out_path ~/Videos/libero_output_pi05/action_entropy_mv4_seed_44 --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 4 --seed 44
# python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_spatial --port 8000 --out_path ~/Videos/libero_output_pi05/action_entropy_mv4_seed_44 --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 4 --seed 44
python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_goal --port 8000 --out_path ~/Videos/libero_output_pi05/action_entropy_mv4_seed_44 --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 4 --seed 44