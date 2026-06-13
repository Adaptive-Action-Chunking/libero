import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os
import cv2
import copy
from action_optimization.help_function import compose_rotations
from collections import deque

def _safe_logdet_cov(X, eps=1e-6):
    X = np.asarray(X)
    N, d = X.shape
    if N <= 1:
        return -np.inf
    Sigma = np.cov(X, rowvar=False)
    Sigma = Sigma + eps * np.eye(d)
    try:
        L = np.linalg.cholesky(Sigma)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            return -np.inf
    return logdet

def gaussian_entropy_from_samples(X):
    X = np.asarray(X)
    N, d  = X.shape
    if N<= 1:
        return 0.0
    logdet = _safe_logdet_cov(X)
    return 0.5 * (d * np.log(2*np.pi*np.e) + logdet)
    
def bernoulli_entropy_from_samples(g):
    g = np.asarray(g).ravel()
    assert len(np.unique(g) <= 2 ), "gripper entropy input should be binary"
    p1 = float(np.clip(np.mean(g), 1e-9, 1-1e-9))
    p0 = 1.0 - p1
    return entropy([p0, p1], base=np.e)




#------------------------ method to select chunk size -----------------------------------------------#

def action_entropy_gausssian_bernoulli(
    pred_action_dict,
    w_pos=1.0,
    w_rot=1.0,
    w_grip=1.0,
    normalized_pose=True
):
    if normalized_pose:
        end_effector_position = pred_action_dict["normalized_action"][:, :, :3]
        end_effector_rotation = pred_action_dict["normalized_action"][:, :, 3:6]
    else:
        end_effector_position = pred_action_dict['action.end_effector_position']
        end_effector_rotation = pred_action_dict["action.end_effector_rotation"]
    
    gripper_close = pred_action_dict["action.gripper_close"]
    gripper_close_p = pred_action_dict["normalized_action"][:, :, 6:7]
    
    N, T, D = end_effector_position.shape
    step_H_pos, step_H_rot, step_H_grip, step_H_total = [], [], [], []
    
    for t in range(T):
        
        H_pos = gaussian_entropy_from_samples(end_effector_position[:, t, :]) 
        H_rot = gaussian_entropy_from_samples(end_effector_rotation[:, t, :])
        H_grip = bernoulli_entropy_from_samples(gripper_close[:, t])
        H_total = w_pos*H_pos + w_rot*H_rot + w_grip*H_grip
        
        step_H_pos.append(H_pos)
        step_H_rot.append(H_rot)
        step_H_grip.append(H_grip)
        step_H_total.append(H_total)
    
    step_H_pos = np.array(step_H_pos)
    step_H_rot = np.array(step_H_rot)
    step_H_grip = np.array(step_H_grip)
    step_H_total = np.array(step_H_total)
    
    H_chunk_mean = [float(np.mean(step_H_total[:k])) for k in range(1, T+1)]
    
    breakdown = {
        "H_pos": step_H_pos,
        "H_rot": step_H_rot,
        "H_grip": step_H_grip,
        "total": step_H_total,
        "chunk_mean": H_chunk_mean,
    }
    return breakdown


def action_variance(pred_action_dict):
    N, T, D = pred_action_dict["normalized_action"].shape
    data = np.asarray(pred_action_dict["normalized_action"][:, :, :7]) # N, T, D
    var_per_dim = data.var(axis=0) # T, D
    
    # select chunk size with variance
    var_total = np.mean(var_per_dim, axis=1)
    var_chunk_mean = [float(np.mean(var_total[:k])) for k in range(1, T+1)]
    
    breakdown = {
        # "var_per_dim": var_per_dim,
        "total": var_total,
        "chunk_mean": var_chunk_mean,
    }
    
    for i in range(var_per_dim.shape[1]):
        breakdown[f"dim_{i}"] = var_per_dim[:, i]
    
    return breakdown


def action_entropy_gaussian_only(pred_action_dict):
    N, T, D = pred_action_dict["normalized_action"].shape
    step_H_total = []
    
    for t in range(T):
        H_total = gaussian_entropy_from_samples(pred_action_dict["normalized_action"][:, t, :7])
        step_H_total.append(H_total)
    
    step_H_total = np.array(step_H_total)
    H_chunk_mean = [float(np.mean(step_H_total[:k])) for k in range(1, T+1)]
    
    breakdown = {
        "total": step_H_total,
        "chunk_mean": H_chunk_mean,
    }
    return breakdown


# separte gaussian + bernoulli
def action_entropy_separate(pred_action_dict):
    
    gripper_close = pred_action_dict["action.gripper_close"]
    N, T, D = pred_action_dict["normalized_action"].shape
    breakdown = {}
    for i in range (6):
        breakdown[f"dim_{i}"] = []
    
    breakdown["H_grip"] = []
    breakdown["total"] = []
    
    for t in range(T):
        H_grip = bernoulli_entropy_from_samples(gripper_close[:, t])
        H_total = H_grip
        breakdown["H_grip"].append(H_grip)
        for i in range (6):
            var = np.var(np.asarray(pred_action_dict["normalized_action"][:, t, i]), ddof=1)
            H = 0.5 * np.log(2 * np.pi * np.e * var)
            H_total += H
            breakdown[f"dim_{i}"].append(H)
        breakdown["total"].append(H_total)
        
    H_total = np.array(breakdown["total"])
    H_chunk_mean = [float(np.mean(H_total[:k])) for k in range(1, T+1)]
    
    breakdown["chunk_mean"] = H_chunk_mean
    return breakdown


# binning
def action_entropy_binning(pred_action_dict, bins=100):
    N, T, D = pred_action_dict["normalized_action"].shape # 20, 16, 32
    gripper_close = pred_action_dict["action.gripper_close"]
    breakdown = {}
    mins = np.zeros(6)-1
    maxs = np.ones(6)
    
    w_grip = 1
    w_other= 1 
    for i in range (6):
        breakdown[f"dim_{i}"] = []
    breakdown["H_grip"] = []
    breakdown["total"] = []
    
    for t in range(T):
        step_data = np.asarray(pred_action_dict["normalized_action"][:, t, :6])
        H_grip = bernoulli_entropy_from_samples(gripper_close[:, t])
        breakdown["H_grip"].append(H_grip)
        H_total = w_grip * H_grip
        for i in range(6):
            x = step_data[:, i]
            edges = np.linspace(mins[i], maxs[i], bins+1)
            counts,_ = np.histogram(x, bins=edges)
            probs = counts[counts>0]/counts.sum()
            H = -np.sum(probs * np.log(probs))
            breakdown[f"dim_{i}"].append(H) 
            H_total += w_other * H
        breakdown["total"].append(H_total)
    
    H_total = np.array(breakdown["total"])
    H_chunk_mean = [float(np.mean(H_total[:k])) for k in range(1, T+1)]
    breakdown["chunk_mean"] = H_chunk_mean
    return breakdown
    

def action_magnitude(pred_action_dict, end_to_end=True, chunk_id=0):
    end_effector_position = np.asarray(pred_action_dict['action.end_effector_position'][chunk_id,])
    end_effector_rotation = np.asarray(pred_action_dict["action.end_effector_rotation"][chunk_id,])
    gripper_close = np.asarray( pred_action_dict["action.gripper_close"][chunk_id,]).astype(int)
    T, D = end_effector_position.shape
    chunk_size = list(range(2, T+1))
    chunk_m = []
    wpos = 1
    wrot = 1
    wgrip = 0.2
    d_pos_k = []
    d_rot_k = []
    for k in chunk_size:
        if end_to_end:
            dpos = np.linalg.norm(np.sum(end_effector_position[:k, ], axis=0))
            drot = np.linalg.norm(compose_rotations(end_effector_rotation[:k, ]))
        else:
            dpos_t = np.linalg.norm(end_effector_position[:k, ], axis=1)
            dpos = np.sum(dpos_t)
            drot_t = np.linalg.norm(end_effector_rotation[:k, ], axis=1)
            drot = np.sum(drot_t)
        gripper_toggle = int(np.any(np.diff(gripper_close[:k, ])))
        m = wpos*dpos + wrot*drot + wgrip*gripper_toggle
        d_pos_k.append(dpos)
        d_rot_k.append(drot)
        chunk_m.append(m)
    # print("d_pos_k:", d_pos_k, )
    # print("d_rot_k:", d_rot_k)
    # print("gripper_close:", gripper_close)
    
    return chunk_m


def find_first_idx(chunk_m, move_th):
    # max_len = len(chunk_m)
    for i, m in enumerate(chunk_m):
        if m > move_th:
            return i+2
    return i+2


def get_elbow_point(vals):
    diffs = np.diff(vals)
    best_k = int(np.argmax(diffs)) + 1
    best_k_entropy = max(best_k, 2) 
    return best_k_entropy

def select_chunk_size(pred_action_dict_input: dict, method, move_th):
    
    pred_action_dict = copy.deepcopy(pred_action_dict_input)
    pred_action_dict["action.gripper_close"] = convert_gripper_to_binary(pred_action_dict["action.gripper_close"])
    
    if method == "gaussian_bernoulli":
        breakdown = action_entropy_gausssian_bernoulli(pred_action_dict)
    elif method == "gaussian_only":
        breakdown = action_entropy_gaussian_only(pred_action_dict)
    elif method == "variance":
        breakdown = action_variance(pred_action_dict)
    elif method == "separate":
        breakdown = action_entropy_separate(pred_action_dict)
    elif method == "binning":
        breakdown = action_entropy_binning(pred_action_dict)
    else:
        raise ValueError
    
    # detect elbow point
    vals = breakdown["chunk_mean"]
    diffs = np.diff(vals)
    best_k = int(np.argmax(diffs)) + 1
    best_k_entropy = max(best_k, 2) 
    # best_k = best_k_entropy
    
    chunk_m = action_magnitude(pred_action_dict)
    min_id =  find_first_idx(chunk_m, move_th)
    best_k = max(min_id, best_k_entropy)
    
    # print("H_pos:", breakdown["H_pos"])
    # print("H_rot:", breakdown["H_rot"])
    # print("H_grip:", breakdown["H_grip"])
    # print("H_total:", breakdown["total"])
    # print("min_id:",min_id)
    # print("best_k_entropy:", best_k_entropy)
    # print("\n")
    
    return best_k, breakdown

def select_chunk_size_mean(pred_action_dict: dict, method, move_th):
    
    if method == "gaussian_bernoulli":
        breakdown = action_entropy_gausssian_bernoulli(pred_action_dict)
    elif method == "gaussian_only":
        breakdown = action_entropy_gaussian_only(pred_action_dict)
    elif method == "variance":
        breakdown = action_variance(pred_action_dict)
    elif method == "separate":
        breakdown = action_entropy_separate(pred_action_dict)
    elif method == "binning":
        breakdown = action_entropy_binning(pred_action_dict)
    else:
        raise ValueError
    
    # detect elbow point
    vals = breakdown["chunk_mean"]
    best_k_entropy = int(np.argmin(vals)) + 1
    # diffs = np.diff(vals)
    # best_k = int(np.argmax(diffs)) + 1
    # best_k_entropy = max(best_k, 2) 
    # best_k = best_k_entropy
    
    chunk_m = action_magnitude(pred_action_dict)
    min_id =  find_first_idx(chunk_m, move_th)
    best_k = max(min_id, best_k_entropy)
    
    # print("H_pos:", breakdown["H_pos"])
    # print("H_rot:", breakdown["H_rot"])
    # print("H_grip:", breakdown["H_grip"])
    # print("H_total:", breakdown["total"])
    print("chunk_mean:", breakdown["chunk_mean"])
    print("min_id:",min_id)
    print("best_k_entropy:", best_k_entropy)
    print("\n")
    
    return best_k, breakdown

def select_chunk_size_mean_without_min_id(pred_action_dict: dict, method, move_th):
    
    if method == "gaussian_bernoulli":
        breakdown = action_entropy_gausssian_bernoulli(pred_action_dict)
    elif method == "gaussian_only":
        breakdown = action_entropy_gaussian_only(pred_action_dict)
    elif method == "variance":
        breakdown = action_variance(pred_action_dict)
    elif method == "separate":
        breakdown = action_entropy_separate(pred_action_dict)
    elif method == "binning":
        breakdown = action_entropy_binning(pred_action_dict)
    else:
        raise ValueError
    
    # detect elbow point
    vals = breakdown["chunk_mean"]
    best_k_entropy = int(np.argmin(vals)) + 1
    # diffs = np.diff(vals)
    # best_k = int(np.argmax(diffs)) + 1
    # best_k_entropy = max(best_k, 2) 
    best_k = best_k_entropy
    
    # chunk_m = action_magnitude(pred_action_dict)
    # min_id =  find_first_idx(chunk_m, move_th)
    # best_k = max(min_id, best_k_entropy)
    
    # print("H_pos:", breakdown["H_pos"])
    # print("H_rot:", breakdown["H_rot"])
    # print("H_grip:", breakdown["H_grip"])
    # print("H_total:", breakdown["total"])
    # print("chunk_mean:", breakdown["chunk_mean"])
    # print("min_id:",min_id)
    # print("best_k_entropy:", best_k_entropy)
    # print("\n")
    
    return best_k, breakdown

#------------------------ method to select chunk id ----------------------------------------#

def select_chunk_id(pred_action_dict: dict, method):
    normalized_action = pred_action_dict["normalized_action"][:, :, :7]
    N, T, D = normalized_action.shape
    flattened = normalized_action.reshape(N, -1) # 20, 16*7
    mean_vector = flattened.mean(axis=0)
    distances = np.linalg.norm(flattened - mean_vector, axis=1)
    best_idx = np.argmin(distances)
    
    return best_idx


# not using now
def adaptive_chunk(pred_action_dict: dict, method):
    chunk_size, breakdown = select_chunk_id(pred_action_dict, method=method)
    chunk_id = select_chunk_id(pred_action_dict)
    processed_action_dict = {}
    for k, v in pred_action_dict.items():
        processed_action_dict[k] = pred_action_dict[k][chunk_id, :chunk_size,]
    return processed_action_dict, breakdown

#----------------------------- tool --------------------------------------#

class EntropyStat:
    def __init__(self, window_size=50, c=0.8):
        self.window_size = window_size
        self.c = c
        self.entropy_history = deque(maxlen=window_size)
        self.mu = None
        self.sigma = None
        self.high_thr = None
        
    def update(self, H_total):
        for h in H_total:
            self.entropy_history.append(h)
            
        self.mu = np.mean(self.entropy_history)
        self.sigma = np.std(self.entropy_history) + 1e-6
        self.high_thr = self.mu + self.c * self.sigma
        
def convert_gripper_to_binary(predictions, threshold=0):
    binary_results = np.array(predictions >= threshold, dtype=int)
    
    return binary_results

##--------------------------------------- visualization ------------------------------------------------------#

Lines_MAP = {
    "gaussian_bernoulli": ["H_pos", "H_rot", "H_grip", "total", "chunk_mean"],
    "gaussian_only": ["total", "chunk_mean"],
    "variance":["dim_0", "dim_1", "dim_2", "dim_3", "dim_4", "dim_5", "dim_6", "total", "chunk_mean"],
    "separate": ["dim_0", "dim_1", "dim_2", "dim_3", "dim_4", "dim_5", "H_grip", "total", "chunk_mean"],
    "binning": ["dim_0", "dim_1", "dim_2", "dim_3", "dim_4", "dim_5", "H_grip", "total", "chunk_mean"],
}


class LiveEntropyPlot:
    def __init__(self, trial_id, method, save_dir=None, visualize=True):
            
        self.visualize = visualize
        self.method = method
        if save_dir:
            episode_dir = os.path.join(save_dir, "plots", f"episode_{trial_id}")
            self.save_dir = episode_dir
            os.makedirs(episode_dir, exist_ok=True)
        else:
            self.save_dir = None
        self.title = f"Uncertainty per step: {method}"

        plt.ion()  # interactive mode
        self.fig, (self.ax, self.ax_img) = plt.subplots(2, 1, figsize=(8,8), 
                                                            gridspec_kw={'height_ratios': [1,1]})

        self.line_best = self.ax.axvline(x=0, color='red', linestyle='--', label='optimal chunk size')
        # self.line_names = ["dim_0", "dim_1", "dim_2", "dim_3", "dim_4", "dim_5", 
        #                    "gripper", "H_pos", "H_rot", "H_grip", "total", "chunk_mean"]
        self.line_names = Lines_MAP[method]

        self.lines = []
        for i in range(len(self.line_names)):
            (ln,) = self.ax.plot([], [], marker='o', label=self.line_names[i], linewidth=2)
            self.lines.append(ln)

        self.ax.set_xlabel("Step (t)")
        self.ax.set_ylabel("Uncertainty")
        self.ax.set_title(self.title)
        self.ax.grid(True)
        # self.ax.legend(loc="upper right", fontsize=9, ncols=1, frameon=False)
        self.ax.legend()
        self.sim_time = self.ax.text(0, 0, "")
        self.img_artist = self.ax_img.imshow([[0]], aspect="auto")
        self.ax_img.axis("off")
        
        if self.visualize:
            self.fig.canvas.draw(); self.fig.canvas.flush_events()

    def update(self, sim_step, data, chunk_size, img, pause=0.05):
        img = copy.deepcopy(img)
        
        self.x = list(range(1, len(data["total"])+1))
        
        for i, line_name in enumerate(self.line_names):
            dim_data = data.get(line_name)
            if dim_data is not None:
                self.lines[i].set_data(self.x, data[line_name])
                
        self.line_best.set_xdata([chunk_size]); self.sim_time.set_text(f"sim_t: {sim_step}")
        
        self.ax.relim(); self.ax.autoscale_view()
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_artist.set_data(rgb_img)
        self.ax_img.draw_artist(self.img_artist)
        
        if self.save_dir:
            filename = os.path.join(self.save_dir, f"entropy_step_{sim_step:03d}_chunk_size_{chunk_size:02d}.png")
            self.fig.canvas.draw(); self.fig.canvas.flush_events()
            self.fig.savefig(filename, dpi=150)
            
        if self.visualize and pause:
            self.fig.canvas.draw(); self.fig.canvas.flush_events()
            plt.pause(pause)  # crucial for GUI refresh
        
    def close(self):
        plt.ioff()
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

