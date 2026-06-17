# Adaptive Action Chunking at Inference-time for Vision-Language-Action Models (LIBERO Evaluation)

This is the official codebase of AAC.

[**[Home page]**](https://lance-lot.github.io/adaptive-chunking.github.io/) &ensp; [**[Paper]**](https://arxiv.org/abs/2604.04161)

-------
## Overview
For a quick review of the implentation of AAC algorithm, see the function select_chunk_size in [action_entropy_v2.py](https://github.com/Adaptive-Action-Chunking/libero/blob/main/action_optimization/action_entropy_v2.py).

This repo is built on offcial [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master) with modifications to support AAC. If you already have LIBERO installed, copy and place the folders (1) action_optimization (2) gr00t_benchmark (3) pi05_benchmark from this repo to your original LIBERO projects. If you don't have LIBERO installed before, follow installation instructions from official [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master). Then add in the above mentioned 3 folders from this repo.

Note:
If you want to align the LIBERO version with our experiments, you can use the command below to specify the LIBERO codebase version.

In the project folder of installed LIBERO:
```sh
git checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
```


## Usage
Our implementation is based on server-client mode. This repo only includes the client part. To run a policy server, we provide the implementation of [GR00T N1.5](https://github.com/Adaptive-Action-Chunking/gr00t-multi-sample) (with modification to support sampling mutiple action chunks in parallel.)

example to run the evaluation client for one task with AAC:
```sh
cd gr00t_benchmark
python inference_client_action_entropy.py --num_trials_per_task 50 --task_suite_name libero_10 --port 8091 --out_path ~/Videos/libero_output/action_entropy --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 3 --seed 42
```

Configure --task_suite_name for different task suites in LIBERO, set --port according to your policy server.

Refer to [gr00t_benchmark/action_entropy.sh](https://github.com/Adaptive-Action-Chunking/libero/blob/main/gr00t_benchmark/action_entropy.sh) for more example scripts.
 
-------
## Citation
```bibtex
@inproceedings{liang2026adaptive,
  title={Adaptive action chunking at inference-time for vision-language-action models},
  author={Liang, Yuanchang and Wang, Xiaobo and Wang, Kai and Wang, Shuo and Peng, Xiaojiang and Chen, Haoyu and Chua, David Kim Huat and Vadakkepat, Prahlad},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20802--20811},
  year={2026}
}
```

