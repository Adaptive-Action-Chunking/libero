# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import argparse
import numpy as np
import robosuite
from robosuite import load_composite_controller_config

# IMPORTANT: you need to import the package to register the environments
import dexmimicgen
from help_function import choose_option, ENV_ROBOTS, PORT_MAP, TASKS, convert_action_bimanual_gripper, pad_image, set_seed
from termcolor import colored
from policy_services import ExternalRobotInferenceClient
import cv2
import pandas as pd


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--render", type=bool, default=True, help="Whether to render the environment")
    
    parser.add_argument("--host", type=str, default="10.72.1.16", help="server address")
    
    parser.add_argument("--port", type=int, default=8081, help="server port")
    
    parser.add_argument("--trials", type=int, default=20, help="number of total trials")
    
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    
    args = parser.parse_args()
    
    env_name = choose_option(
            TASKS, "task", default="TwoArmThreading", show_keys=True
        )  
    # env_name = "TwoArmThreePieceAssembly"
    
    num_trials = args.trials
    success_count = 0  # global success count
    if args.seed:
        set_seed(seed=42)
    
    for trial_id in range(num_trials):
        print(f"trial {trial_id}")
        # Create dict to hold options that will be passed to env creation call
        env_kwargs = {
            "env_name": env_name,
            "robots": ENV_ROBOTS[env_name],
            "controller_configs": load_composite_controller_config(
                robot=ENV_ROBOTS[env_name][0]
            ),
            "has_renderer": args.render,
            "has_offscreen_renderer": True,
            "ignore_done": False,
            "use_camera_obs": True,
            "control_freq": 20,
            "horizon": 720,
            "camera_heights": 164,
            "camera_widths": 256,
            "camera_names": ["agentview", "robot0_eye_in_hand", 'robot1_eye_in_hand'] # Available "camera" names = ('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand', 'robot1_robotview', 'robot1_eye_in_hand').
        }
        # initialize the task
        env = robosuite.make(**env_kwargs,)
        env.reset()
        
        zero_action = np.zeros(*env.action_spec[0].shape)
        # do a dummy step thru base env to initalize things, but don't record the step
        obs, reward, done, _  = env.step(zero_action)
        if args.render:
            env.render()

        # initialize policy client
        policy = ExternalRobotInferenceClient(host=args.host, port=PORT_MAP[args.port], timeout_ms=5000)
        
        trial_success = False
        end_episode = False
        task_completion_hold_count = -1 
        
        lang = TASKS[env_name]

        while end_episode==False and trial_success==False:
            # collect observations
            
            obs_dict = {}
            obs_dict['video.right_wrist_view'] = pad_image(obs["robot0_eye_in_hand_image"])[None, ...]
            obs_dict['video.left_wrist_view'] = pad_image(obs["robot1_eye_in_hand_image"])[None, ...]
            obs_dict['video.front_view'] = pad_image(obs["agentview_image"])[None, ...]
            
            obs_dict["state.right_arm_eef_pos"] = obs["robot0_eef_pos"][None, ...]
            obs_dict["state.right_arm_eef_quat"] = obs["robot0_eef_quat"][None, ...]
            obs_dict["state.right_gripper_qpos"] = obs["robot0_gripper_qpos"][None, ...]
            
            obs_dict["state.left_arm_eef_pos"] = obs["robot1_eef_pos"][None, ...]
            obs_dict["state.left_arm_eef_quat"] = obs["robot1_eef_quat"][None, ...]
            obs_dict["state.left_gripper_qpos"] = obs["robot1_gripper_qpos"][None, ...]
            
            obs_dict["annotation.human.action.task_description"] = [lang]
            
            pred_action_dict = policy.get_action(obs_dict)
            
            chunk_size = pred_action_dict["action.right_gripper_close"].shape[0]
            
            # execute action chunk
            for t in range(chunk_size):
                action = convert_action_bimanual_gripper(pred_action_dict, t, env)
                # execute action step
                obs, reward, done, info = env.step(action)
                # visualization 
                cam_names = env_kwargs["camera_names"]
                if args.render:
                    env.render()
                    for _, show_cam in enumerate(cam_names):
                        image_array = obs[show_cam + "_image"]
                        cv2.putText(image_array, "TOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        bgr_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                        cv2.imshow(f"{show_cam}", pad_image(bgr_img))
                    cv2.waitKey(1)
                
                # state machine to check for having a success for 10 consecutive timesteps
                if env._check_success():
                    if task_completion_hold_count > 0:
                        task_completion_hold_count -= 1  # latched state, decrement count
                    else:
                        task_completion_hold_count = 10  # reset count on first success timestep
                else:
                    task_completion_hold_count = -1  # null the counter if there's no success
                
                # next trial if this episode succeeds
                if task_completion_hold_count == 0:
                    trial_success = True # exit this trial
                    success_count += 1
                    print(f"Task success! current success rate: {success_count/(trial_id+1):.2f}")
                    break # exit action chunk execution
                
                if done:
                    end_episode = True # exit this trial
                    print("End of episode.")
                    break # exit action chunk execution
            
        env.close()
    
    # print success rate after all trials
    log_content = f"success_count:{success_count}, num_trials:{num_trials}, success_rate:{success_count/num_trials}"
    print(log_content)