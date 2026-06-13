# lerobot v0.3.3
# cv2 opencv-python-headless       4.12.0.88
# h5py                         3.14.0

import shutil
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import cv2
import h5py
import os
import glob

# Input
DATA_DIR = '/home/sangfor/code/lyc/datasets/aloha_mobile_dummy'

# OUTPUT
REPO_NAME = "/home/sangfor/code/lyc/datasets/pick_banana_demo"  # Name of the output dataset, also used for the Hugging Face Hub
TASK_DESCRIPTION = "Pick up the banana and place it in the basket."
USE_VIDEO = True
FPS = 30

def load_hdf5(dataset_path):
    # dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        if 'effort' in root.keys():
            effort = root['/observations/effort'][()]
        else:
            effort = None
        action = root['/action'][()]
        base_action = root['/base_action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        if compressed:
            compress_len = root['/compress_len'][()]

    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            # un-pad and uncompress
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list): # [:1000] to save memory
                image_len = int(compress_len[cam_id, frame_id])
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                image_list.append(image)
            image_dict[cam_name] = image_list

    return qpos, qvel, effort, action, base_action, image_dict




def main(data_dir: str, out_repo_name: str):
    # Clean up any existing dataset in the output directory
    # output_path = REPO_NAME
    # output_path = HF_LEROBOT_HOME / REPO_NAME
    # if output_path.exists():
    #     shutil.rmtree(output_path)
    if os.path.exists(out_repo_name):
        shutil.rmtree(out_repo_name)
    
    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dtype = "video" if USE_VIDEO else "image"
    dataset = LeRobotDataset.create(
        repo_id=out_repo_name,
        robot_type="cobot magic",
        fps=FPS,
        features={
            "observation.images.cam_high": {
                "dtype": dtype,
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.cam_left_wrist": {
                "dtype": dtype,
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.cam_right_wrist": {
                "dtype": dtype,
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
        use_videos=USE_VIDEO
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    episode_list = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    for ep_idx, episode_file in enumerate(episode_list):
        qpos, qvel, effort, action, base_action, image_dict = load_hdf5(episode_file)

        print(f'episode_{ep_idx}')
        len_episode = qpos.shape[0]
        # pass
        for step_idx in range(len_episode):
        # for step in episode["steps"].as_numpy_iterator():
            dataset.add_frame(
                {
                    "observation.images.cam_high": image_dict["cam_high"][step_idx],
                    "observation.images.cam_left_wrist": image_dict["cam_left_wrist"][step_idx],
                    "observation.images.cam_right_wrist": image_dict["cam_right_wrist"][step_idx],
                    "observation.state": qpos[step_idx],
                    "action": action[step_idx],
                },
                task=TASK_DESCRIPTION,
            )
        dataset.save_episode()


if __name__ == "__main__":
    main(data_dir = DATA_DIR, out_repo_name = REPO_NAME)
