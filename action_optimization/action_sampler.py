import numpy as np

def euclidean_distance(src, tar, reduction='mean'):
    # N, T, D
    diff = src - tar
    dist = np.linalg.norm(diff, axis=-1)
    if reduction == 'mean':
        return dist.mean(axis=-1)
    elif reduction == 'none':
        return dist


def backward(curr_action_overlap, last_action_overlap, num_sample, beta=0.99):
    dist_raw = euclidean_distance(src=curr_action_overlap, tar=last_action_overlap, reduction="none")
        
    weights = np.array([beta**i for i in range(last_action_overlap.shape[1])])
    weights = weights / weights.sum()
    dist_weighted = dist_raw * weights.reshape(1, 1, last_action_overlap.shape[1])
    dist_sum = dist_weighted.sum(axis=2)
    cross_index = np.argsort(dist_sum, axis=1)
    best_ids = cross_index[0, :num_sample]
    
    return best_ids, dist_sum[0, best_ids]

def forward(curr_action_overlap, curr_action_overlap_weak, K=3, beta=0.99):
    dist_raw = euclidean_distance(src=curr_action_overlap, tar=curr_action_overlap_weak, reduction="none")
        
    weights = np.array([beta**i for i in range(curr_action_overlap_weak.shape[1])])
    weights = weights / weights.sum()
    dist_weighted = dist_raw * weights.reshape(1, 1, curr_action_overlap_weak.shape[1])
    dist_weak_sum = dist_weighted.sum(axis=2)
    # cross_index = np.argsort(dist_strong_sum, axis=1)
    # best_ids = cross_index[0][0]
    
    return dist_weak_sum


#------------------------ method to select chunk id ----------------------------------------#

def select_chunk_id(pred_action_dict: dict, method, last_action_dict, pred_chunk_size, pred_action_dict_weak=None, beta=0.99):
    
    if method == "mean":
        normalized_action = pred_action_dict["normalized_action"][:, :, :7]
        N, T, D = normalized_action.shape
        flattened = normalized_action.reshape(N, -1) # 20, 16*7
        mean_vector = flattened.mean(axis=0)
        distances = np.linalg.norm(flattened - mean_vector, axis=1)
        best_idx = np.argmin(distances)
    
    elif method == "backward":
        use_predicted_chunk_size_only = True # whether to use all rest unexecuted actions from last prediction 
        if not last_action_dict:
            return 0 # first inference
        
        full_action_horizon = pred_action_dict["normalized_action"].shape[1]
        last_chunk_size = last_action_dict["chunk_size"]
        
        if last_chunk_size == full_action_horizon:
            return 0 # no overlap actions, use the first action by default
        
        if use_predicted_chunk_size_only:
            last_action_overlap = last_action_dict["normalized_action"][:, last_chunk_size : last_chunk_size + pred_chunk_size, :7]
        else:
            last_action_overlap = last_action_dict["normalized_action"][:, last_chunk_size:, :7]
        
        curr_action_overlap = pred_action_dict["normalized_action"][:, :last_action_overlap.shape[1], :7]
        
        dist_raw = euclidean_distance(src=curr_action_overlap, tar=last_action_overlap, reduction="none")
        
        weights = np.array([beta**i for i in range(last_action_overlap.shape[1])])
        weights = weights / weights.sum()
        dist_weighted = dist_raw * weights.reshape(1, 1, last_action_overlap.shape[1])
        dist_strong_sum = dist_weighted.sum(axis=2)
        cross_index = np.argsort(dist_strong_sum, axis=1)
        best_idx = cross_index[0][0]
        
    elif method == "BID":

        
        use_predicted_chunk_size_only = False # whether to use all rest unexecuted actions from last prediction 
        if not last_action_dict:
            return 0 # first inference
        
        full_action_horizon = pred_action_dict["normalized_action"].shape[1]
        last_chunk_size = last_action_dict["chunk_size"]
        
        if last_chunk_size == full_action_horizon:
            return 0 # no overlap actions, use the first chunk by default
        
        if use_predicted_chunk_size_only:
            last_action_overlap = last_action_dict["normalized_action"][:, last_chunk_size : last_chunk_size + pred_chunk_size, :7]
        else:
            last_action_overlap = last_action_dict["normalized_action"][:, last_chunk_size:, :7]
        
        curr_action_overlap = pred_action_dict["normalized_action"][:, :last_action_overlap.shape[1], :7]
        curr_action_overlap_weak = pred_action_dict_weak["normalized_action"][:, :last_action_overlap.shape[1], :7]

        # filter actions that close to prior prediction
        num_mode = 3
        num_sample = curr_action_overlap.shape[0] // num_mode 
        back_ids_strong, dist_pior = backward(curr_action_overlap, last_action_overlap, num_sample=num_sample)
        back_ids_weak, _ = backward(curr_action_overlap_weak, last_action_overlap, num_sample=num_sample)
        
        action_filtered_strong = curr_action_overlap[back_ids_strong, :, :]
        action_filtered_weak = curr_action_overlap_weak[back_ids_strong, :, :]
        
        # distance to strong predictions
        dist_pos_raw = euclidean_distance(src=action_filtered_strong, tar=action_filtered_strong, reduction="mean")
        top_k = num_sample // 2 +1
        all_ids = np.argsort(dist_pos_raw, axis=-1)
        topk_ids = all_ids[:, :top_k]
        values = dist_pos_raw[:, topk_ids]
        dist_pos = values.mean(axis=-1)
        
        
        forward_dist = forward()
        
        # dist_raw = euclidean_distance(src=curr_action_overlap, tar=last_action_overlap, reduction="none")
        
        # weights = np.array([beta**i for i in range(last_action_overlap.shape[1])])
        # weights = weights / weights.sum()
        # dist_weighted = dist_raw * weights.reshape(1, 1, last_action_overlap.shape[1])
        # dist_strong_sum = dist_weighted.sum(axis=2)
        # cross_index = np.argsort(dist_strong_sum, axis=1)
        # best_idx = cross_index[0][0]
    
    else:
        raise ValueError
    
    return best_idx

