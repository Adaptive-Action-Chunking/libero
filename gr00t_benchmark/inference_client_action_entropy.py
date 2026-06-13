import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, DemoRenderEnv
import numpy as np
import tqdm
import tyro
import pandas as pd
import cv2
import argparse
from benchmark_scripts.policy_services import ExternalRobotInferenceClient
import os
from datetime import datetime
import json
from action_optimization.action_entropy_v2 import  LiveEntropyPlot, select_chunk_size
from action_optimization.help_function import merge_cam_view
from action_optimization.action_sampler import select_chunk_id
import copy
import csv
from results_logger import ResultLogger

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

PORT_MAP = {
    8080: 59049,
    8081: 2944,
    8082: 64098,
    8083: 2645,
    8084: 34738,
    8086: 56973,
    8088: 32953,
    8089: 51198,
    8090: 38094,
    8091: 60390,
    8092: 58224,
    8093: 51290,
    8094: 53537,
    8095: 37536,
    8096: 62907
    }
    
def main():

    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser()

    # debug
    parser.add_argument("--debug", type=bool, default=False, help="debug mode")
    parser.add_argument("--render", type=bool, default=False, help="Whether to render the environment")
    parser.add_argument("--vis_plots", type=bool, default=False, help="visualize action entropy plots")
    
    # experiment
    parser.add_argument("--out_path", type=str, default="/home/sangfor/Videos/libero_output/action_entropy_test_debug", help="saved video results")
    parser.add_argument("--save_videos", type=bool, default=True, help="whether to save video results")
    parser.add_argument("--save_plots", type=bool, default=False, help="whether to save plots")
    parser.add_argument("--save_logs", type=bool, default=True, help="whether to save csv logs")
    
    # algorithm related
    parser.add_argument("--chunk_size_selector", type=str, default="gaussian_bernoulli", help="whether to save video results")
    # gaussian_bernoulli, gaussian_only, variance, separate, binning, fixed_{int}
    parser.add_argument("--chunk_id_selector", type=str, default="0", help="method to choose chunk id")
    # "backward", "0" 
    parser.add_argument("--move_th", type=float, default=3.0, help="Minimum movement threshold")
    
    # server config
    parser.add_argument("--host", type=str, default="10.72.1.16", help="server address")
    parser.add_argument("--port", type=int, default=8091, help="server port")
    
    # simulation config
    parser.add_argument("--num_trials_per_task", type=int, default=50, help="number of trials per task")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--task_suite_name", type=str, default="libero_10", help="Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90")
    parser.add_argument("--num_steps_wait", type=int, default=10, help="Number of steps to wait for objects to stabilize in sim")
    
    cam_names = ["agentview", "robot0_eye_in_hand"]
    
    args = parser.parse_args()
    
    if args.debug:
        args.save_videos = False
        args.save_plots = False
    
    
    # Set random seed
    np.random.seed(args.seed)
    
    now = datetime.now()
    args.out_path = os.path.join(args.out_path + "_" + args.chunk_size_selector + "_" + args.chunk_id_selector, args.task_suite_name + "_" + now.strftime("%Y%m%d_%H%M%S"))
    video_out_path = os.path.join(args.out_path, "videos")
    if args.save_videos:
        os.makedirs(video_out_path, exist_ok=True)
    
        args_dict = vars(args)
        with open(os.path.join(args.out_path, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, ensure_ascii=False, indent=4)
    
    if args.save_logs:
        logger = ResultLogger(output_folder=args.out_path, test_name="results_log_", fieldnames=["task_id", "trial_id", "instruction", "success", "end_time"])

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # initialize policy client
    policy = ExternalRobotInferenceClient(host=args.host, port=PORT_MAP[args.port], timeout_ms=5000)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    chunk_size_stats = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
    
        # Get task 
        # task_id = 8 # DEBUG only 
        task = task_suite.get_task(task_id) 

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_segment = task_description.replace(" ", "_")
        video_out_path = os.path.join(args.out_path, f"videos/{task_segment}")
        os.makedirs(video_out_path, exist_ok=True)
        
        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            
            # episode_idx=1 # DEBUG ONLY
            
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            
            logging.info(f"Trial {episode_idx}...")
            
            
            if args.save_plots or args.vis_plots:
                path = args.out_path if args.save_plots else None
                entropy_plotter = LiveEntropyPlot(episode_idx, method=args.chunk_size_selector, save_dir=path, visualize=args.vis_plots)
            
            trial_success = False
            end_episode = False
            last_action_dict = None
            inference_id = 0
            
            while end_episode==False and trial_success==False:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                
                # ----------------------- setp 1 collect observation  ------------------------------------#
        
                obs_dict = {}
                obs_dict['video.wrist_view'] = np.copy( obs["robot0_eye_in_hand_image"][::-1, ::-1, :])[None, ...]
                obs_dict['video.front_view'] = np.copy( obs["agentview_image"][::-1, ::-1, :]) [None, ...]
                obs_dict["state.end_effector_position"] = obs["robot0_eef_pos"][None, ...]
                obs_dict["state.end_effector_rotation"] = _quat2axisangle(obs["robot0_eef_quat"])[None, ...]
                obs_dict["state.gripper_qpos"] = obs["robot0_gripper_qpos"][None, ...]
                obs_dict["annotation.human.action.task_description"] = [task_description]
                
                # ------------------------setp 2 get action ----------------------------------------------- #
                
                pred_action_dict = policy.get_action(obs_dict)
                pred_action_dict_original = copy.deepcopy(pred_action_dict)
                
                # ----------------------- setp 3 process action  -------------------------------------------#
                # select chunk size
                if "fixed" in args.chunk_size_selector:
                    chunk_size = int(args.chunk_size_selector.split("_")[-1])
                    for k, v in pred_action_dict.items():
                        pred_action_dict[k] = pred_action_dict[k][:, :chunk_size]
                else:
                    chunk_size, breakdown = select_chunk_size(pred_action_dict, method=args.chunk_size_selector, move_th=args.move_th)
                    for k, v in pred_action_dict.items():
                        pred_action_dict[k] = pred_action_dict[k][:, :chunk_size]
                    
                    if args.save_plots or args.vis_plots:
                        merge_view = merge_cam_view( obs, cam_names, episode_idx, t, task_description, chunk_size)
                        entropy_plotter.update(t, breakdown, chunk_size, img=merge_view) 
                
                # slelect chunk id
                if args.chunk_id_selector == "backward":
                    chunk_id = select_chunk_id(pred_action_dict_original, method=args.chunk_id_selector, last_action_dict=last_action_dict, pred_chunk_szie=chunk_size)
                    for k, v in pred_action_dict.items():
                        pred_action_dict[k] = pred_action_dict[k][chunk_id, :]
                else:
                    chunk_id = 0 # select the first chunk by default
                    for k, v in pred_action_dict.items():
                        pred_action_dict[k] = pred_action_dict[k][chunk_id, :]
                
                last_action_dict = pred_action_dict_original
                for k, v in last_action_dict.items():
                    last_action_dict[k] = last_action_dict[k][chunk_id:chunk_id+1, :]
                last_action_dict["chunk_size"] = chunk_size 
                
                chunk_size_stats.append({
                "task_id": task_id,
                "trial_id": episode_idx,
                "inference_id": inference_id,
                "sim_step": t,
                "chunk_size": chunk_size
                })
                inference_id += 1
                
                # ----------------------- setp 4 execute action in simulation -----------------------------#
                chunk_to_execute = pred_action_dict[next(iter(pred_action_dict))].shape[0]
                for a_t in range(chunk_to_execute):
                    # convert pred_action_dict to action
                    action_dict = {}
                    action_dict["right"] = np.concatenate( (pred_action_dict["action.end_effector_position"],pred_action_dict["action.end_effector_rotation"]), axis = 1)[a_t].tolist()
                    action_dict["right_gripper"] = [(pred_action_dict["action.gripper_close"][a_t]-0.5)*2]
                    # action_dict["right_gripper"] = [pred_action_dict["action.gripper_close"][a_t]]
                    
                    
                    action = np.concatenate([action_dict["right"], action_dict["right_gripper"]], axis=0)
                    # execute action in environment
                    obs, reward, done, info = env.step(action)
                    t += 1
                    
                    # visualization                    
                    if args.save_videos or args.render:
                        # merge_view_list = []
                        # cam_names = ["agentview", "robot0_eye_in_hand"]
                        # for _, show_cam in enumerate(cam_names):
                        #     image_array = np.copy(obs[show_cam + "_image"][::-1, ::-1, :]) # flip camera up side down, left to right
                        #     cv2.putText(image_array, "TOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        #     bgr_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                        #     merge_view_list.append(bgr_img)
                        # merge_view = np.concatenate(merge_view_list, axis=1)
                        merge_view = merge_cam_view(obs, cam_names, episode_idx, t, task_description, chunk_size)
                        if args.save_videos:
                            merge_view_RGB = cv2.cvtColor(merge_view, cv2.COLOR_BGR2RGB)
                            replay_images.append(merge_view_RGB)
                        if args.render:
                            env.env.render()
                            cv2.imshow("video", merge_view)
                            if cv2.waitKey(1) & 0xFF == 27:
                                print("exit.")
                                env.close()
                                exit()
                    if done:
                        print("task scuess!")
                        trial_success = True
                        task_successes += 1
                        total_successes += 1
                        break
                    
                    if t >= max_steps:
                        end_episode = True # exit this trial
                        print("End of episode.")
                        break # exit action chunk execution
                
            # trial finish 
            if args.save_plots or args.vis_plots:
                entropy_plotter.close()
                    
            task_episodes += 1
            total_episodes += 1
            
            # Save a replay video of the episode
            if args.save_videos:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(video_out_path) / f"rollout_{task_segment}_trial_{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            episode_success = 1 if trial_success else 0
            logger.log(task_id=task_id, trial_id=episode_idx, instruction=task_description, success=episode_success, end_time=t)
            
            # Log current episode results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            
        env.close()
        # Log final task results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        txt_file = os.path.join(args.out_path, "success_rate.txt")
        if args.save_logs:
            with open(txt_file, "a", encoding="utf-8") as file:
                content = f"task {task_id}:{task_description}, success count:{task_successes}, num_trials:{task_episodes}, success_rate:{task_successes/task_episodes}\n"
                file.write(content)
        
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    
    if args.save_logs:
        with open(txt_file, "a", encoding="utf-8") as file:
                content = f"Total success count:{total_successes}, total num_trials:{total_episodes}, total success_rate:{total_successes/total_episodes}\n"
                file.write("\n")
                file.write(content)
        logger.save()
        
        chunk_size_csv = os.path.join(args.out_path, "chunk_size_log.csv")
        with open(chunk_size_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task_id", "trial_id", "inference_id", "sim_step", "chunk_size"])
            writer.writeheader()
            writer.writerows(chunk_size_stats)


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    # env = OffScreenRenderEnv(**env_args)
    env = DemoRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


if __name__ == "__main__":
    main()
