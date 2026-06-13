import collections
import dataclasses
import logging
import math
import pathlib
import json

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

def transform_action_to_base_frame(actions_target, R_base2cam, source_frame="image"):
    """
    将 target_frame (image 或 camera) 下的 action 逆向转换回 base 坐标系
    """
    delta_pos_cam = actions_target[:, :3].copy()
    delta_rotvec_cam = actions_target[:, 3:6].copy()
    gripper_states = actions_target[:, 6:].copy()
    
    # 1. 恢复 Image 坐标系的翻转 (如果是从 image 坐标系过来的)
    if source_frame == "image":
        delta_pos_cam[:, 1:3] *= -1.0
        delta_rotvec_cam[:, 1:3] *= -1.0
        
    # 2. 从 Camera 转回 Base 
    # 正向是右乘 R.T，逆向则是右乘 R
    delta_pos_base = np.dot(delta_pos_cam, R_base2cam)
    delta_rotvec_base = np.dot(delta_rotvec_cam, R_base2cam)
    
    transformed_actions = np.hstack([delta_pos_base, delta_rotvec_base, gripper_states])
    return transformed_actions.astype(np.float32)

@dataclasses.dataclass
class Args:
    resize_size: int = 224
    task_suite_name: str = "libero_object"  
    num_steps_wait: int = 10  
    num_trials_per_task: int = 50  
    seed: int = 7  
    task_id: int = 0
    render: bool = True
    
    # 新增参数
    source_frame: str = "image"  # 新数据集的 action 坐标系 ("image" 或 "camera")
    extrinsics_path: str = "./libero_two_camera_extrinsics.json"
    dataset_path: str = "/home/sangfor/datasets/binhng/libero_object_lerobot_mask_depth_converted_to_image/data/chunk-000/episode_000002.parquet"

def main(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    logging.info(f"Task suite: {args.task_suite_name}")

    # 获取任务及描述
    task = task_suite.get_task(args.task_id)
    task_description = task.language
    print("task_description:", task_description)

    # ---------------------------------------------------------
    # 1. 加载外参矩阵 R_base2cam
    # ---------------------------------------------------------
    with open(args.extrinsics_path, "r", encoding="utf-8") as f:
        extrinsics_dict = json.load(f)
        
    R_base2cam = None
    for suite, tasks in extrinsics_dict.items():
        if task_description in tasks:
            R_base2cam = np.array(tasks[task_description]["R_base2cam"])
            break
            
    if R_base2cam is None:
        raise ValueError(f"在 {args.extrinsics_path} 中找不到任务 '{task_description}' 的外参！")

    # ---------------------------------------------------------
    # 2. 加载新数据集的 GT Actions 并逆向转换
    # ---------------------------------------------------------
    df = pd.read_parquet(args.dataset_path)
    actions_loaded = np.stack(df["action"].to_numpy())
    
    # 如果 LeRobot 保存的 gripper 是 [0, 1]，转换回 MuJoCo 的 [-1, 1]
    actions_loaded[:, -1:] = (actions_loaded[:, -1:] - 0.5) * 2

    # 执行逆向转换：Image/Camera -> Base
    actions_base = transform_action_to_base_frame(actions_loaded, R_base2cam, source_frame=args.source_frame)

    # ---------------------------------------------------------
    # 3. 仿真环境执行
    # ---------------------------------------------------------
    initial_states = task_suite.get_task_init_states(args.task_id)
    env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    env.reset()
    obs = env.set_init_state(initial_states[0])

    t = 0
    task_successes = 0
    total_successes = 0

    while t < len(actions_base) + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue
        
        # 将转换回 Base 坐标系的 action 喂给环境
        obs, reward, done, info = env.step(actions_base[t - args.num_steps_wait])
        
        if args.render:
            env.env.render()
            cam_names = ["agentview"]
            for _, show_cam in enumerate(cam_names):
                image_array = np.copy(obs[show_cam + "_image"][::-1, ::-1, :]) 
                cv2.putText(image_array, "TOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                bgr_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"{show_cam}", bgr_img)
            cv2.waitKey(1)
        
        if done:
            task_successes += 1
            total_successes += 1
            print(f"✅ 任务执行成功！在第 {t} 步完成。")
            break
        t += 1
        
    if not done:
        print("❌ 轨迹执行完毕，但任务未触发成功判定。")
        
    env.close()

def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = DemoRenderEnv(**env_args)
    env.seed(seed)  
    return env, task_description

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)