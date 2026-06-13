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
from policy_services import ExternalRobotInferenceClient

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
    }
    
def main():

    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--render", type=bool, default=True, help="Whether to render the environment")
    
    parser.add_argument("--host", type=str, default="10.72.1.16", help="server address")
    
    parser.add_argument("--port", type=int, default=8089, help="server port")
    
    parser.add_argument("--num_trials_per_task", type=int, default=50, help="number of trials per task")
    
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    
    parser.add_argument("--task_suite_name", type=str, default="libero_goal", help="Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90")
    
    parser.add_argument("--num_steps_wait", type=int, default=10, help="Number of steps to wait for objects to stabilize i n sim")
    
    parser.add_argument("--video_out_path", type=str, default="/home/sangfor/Videos/libero_ouput", help="saved video results")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)

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
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
    
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        
        if task_description != "put the bowl on the plate":
            continue
        
        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[0])

            # Setup
            t = 0
            replay_images = []
            
            logging.info(f"Trial {episode_idx}...")
            
            trial_success = False
            end_episode = False
            while end_episode==False and trial_success==False:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                
                # collect observations
        
                obs_dict = {}
                obs_dict['video.wrist_view'] = np.copy( obs["robot0_eye_in_hand_image"][::-1, ::-1, :])[None, ...]
                obs_dict['video.front_view'] = np.copy( obs["agentview_image"][::-1, ::-1, :]) [None, ...]
                
                obs_dict["state.end_effector_position"] = obs["robot0_eef_pos"][None, ...]
                obs_dict["state.end_effector_rotation"] = _quat2axisangle(obs["robot0_eef_quat"])[None, ...]
                obs_dict["state.gripper_qpos"] = obs["robot0_gripper_qpos"][None, ...]
                
                obs_dict["annotation.human.action.task_description"] = [task_description]
                
                pred_action_dict = policy.get_action(obs_dict)
                
                if len(pred_action_dict[next(iter(pred_action_dict))].shape) == 3:
                    B, T, D = pred_action_dict[next(iter(pred_action_dict))].shape
                    chunk_id=0
                    for k, v in pred_action_dict.items():
                        pred_action_dict[k] = pred_action_dict[k][chunk_id, ]
                else:
                    T, D = pred_action_dict[next(iter(pred_action_dict))].shape
        
                chunk_size = T
                
                # execute action chunk
                for a_t in range(chunk_size):
                    # convert pred_action_dict to action
                    action_dict = {}
                    action_dict["right"] = np.concatenate( (pred_action_dict["action.end_effector_position"],pred_action_dict["action.end_effector_rotation"]), axis = 1)[a_t].tolist()
                    # action_dict["right_gripper"] = [(action_dict["action.gripper_close"][a_t]-0.5)*2]
                    action_dict["right_gripper"] = [pred_action_dict["action.gripper_close"][a_t]]
                    
                    # active_robot = env.env.robots[0]
                    # action = active_robot.create_action_vector(action_dict)
                    
                    action = np.concatenate([action_dict["right"], action_dict["right_gripper"]], axis=0)
                    # execute action in environment
                    obs, reward, done, info = env.step(action)
                    
                    # visualization
                    if args.render:
                        env.env.render()
                        cam_names = ["agentview", "robot0_eye_in_hand"]
                        for _, show_cam in enumerate(cam_names):
                            image_array = np.copy(obs[show_cam + "_image"][::-1, ::-1, :]) # flip camera up side down, left to right
                            cv2.putText(image_array, "TOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            bgr_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                            cv2.imshow(f"{show_cam}", bgr_img)
                        cv2.waitKey(1)
                    
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
                    
                    a_t += 1
            
            task_episodes += 1
            total_episodes += 1
            
            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            
            # Log current episode results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            
        env.close()
        # Log final task results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


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
