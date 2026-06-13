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

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

@dataclasses.dataclass
class Args:
    resize_size: int = 224
    
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    seed: int = 7  # Random Seed (for reproducibility)
    
    task_id: int = 0
    render: bool = True
    

def main(args: Args) -> None:
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

    # get gt actions
    # path = "/home/sangfor/code/lyc/datasets/libero_lerobot_spatial/data/chunk-000/episode_000001.parquet"
    path = "/home/sangfor/datasets/binhng/libero_object_lerobot_mask_depth/data/chunk-000/episode_000002.parquet"
    df = pd.read_parquet(path)
    actions = np.stack(df["action"].to_numpy())
    actions[:, -1:] =  (actions[:, -1:]-0.5)*2

    # Get task
    task = task_suite.get_task(args.task_id)

    # Get default LIBERO initial states
    initial_states = task_suite.get_task_init_states(args.task_id)

    # Initialize LIBERO environment and task description
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    print("task_description:", task_description)

    # Reset environment
    env.reset()

    # Set initial states
    obs = env.set_init_state(initial_states[0])

    # Setup
    t = 0

    while t < len(actions) + args.num_steps_wait:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue
            
            # Execute action in environment
            obs, reward, done, info = env.step(actions[t-args.num_steps_wait])
            
            if args.render:
                env.env.render()
                cam_names = ["agentview"]
                for _, show_cam in enumerate(cam_names):
                    image_array = np.copy(obs[show_cam + "_image"][::-1, ::-1, :]) # flip camera up side down, left to right
                    cv2.putText(image_array, "TOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    bgr_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"{show_cam}", bgr_img)
                cv2.waitKey(1)
            
            if done:
                task_successes += 1
                total_successes += 1
                break
            t += 1
    env.close()


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
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
