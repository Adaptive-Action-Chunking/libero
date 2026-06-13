import numpy as np

def euclidean_distance(src, tar, reduction='mean'):
    """
    计算欧氏距离，支持广播
    src: [N, T, D] or [N, 1, T, D]
    tar: [M, T, D] or [1, M, T, D]
    Output: 
       reduction='mean' -> [N] or [N, M]
       reduction='none' -> [N, T] or [N, M, T]
    """
    # diff: [N, M, T, D] (当发生广播时)
    diff = src - tar
    
    # 在最后一个维度 (Action Dim) 上求范数 -> [N, M, T]
    dist = np.linalg.norm(diff, axis=-1)
    
    if reduction == 'mean':
        # 在时间维度 (T) 上求平均 -> [N, M]
        return dist.mean(axis=-1)
    elif reduction == 'none':
        return dist
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def backward(curr_action_overlap, last_action_overlap, num_sample, beta=0.99):
    """
    Backward Coherence: 筛选与上一帧动作重叠部分最连贯的样本
    curr: [N, T, 7]
    last: [1, T, 7]
    """
    # 1. 计算每个时间步的距离: [N, T]
    dist_raw = euclidean_distance(src=curr_action_overlap, tar=last_action_overlap, reduction="none")
    
    # 2. 生成时间衰减权重: [T]
    T = last_action_overlap.shape[1]
    weights = np.array([beta**i for i in range(T)])
    weights = weights / weights.sum() # 归一化
    
    # 3. 加权求和: [N, T] * [1, T] -> sum -> [N]
    # 注意：这里 weights reshape 为 (1, T) 以便广播
    dist_weighted = dist_raw * weights.reshape(1, T)
    dist_sum = dist_weighted.sum(axis=1) 
    
    # 4. 排序并筛选 TopK
    # argsort 默认是从小到大，取前 num_sample 个最小的
    cross_index = np.argsort(dist_sum) 
    best_ids = cross_index[:num_sample]
    
    # 返回: (筛选后的索引, 对应的距离分数)
    return best_ids, dist_sum[best_ids]


def select_chunk_id(pred_action_dict: dict, method, last_action_dict, pred_chunk_size, pred_action_dict_weak=None, beta=0.99):
    """
    BID (Bidirectional Decoding) Selection Logic for Inference
    """
    
    if method == "mean":
        # 原始的 Mean 逻辑保持不变
        normalized_action = pred_action_dict["normalized_action"][:, :, :7]
        N, T, D = normalized_action.shape
        flattened = normalized_action.reshape(N, -1) 
        mean_vector = flattened.mean(axis=0)
        distances = np.linalg.norm(flattened - mean_vector, axis=1)
        best_idx = np.argmin(distances)
        return best_idx
        
    elif method == "BID":
        # === 准备数据 ===
        # 获取当前 Strong Policy 的全量预测 [N, T, D]
        # 强制取前7维，防止 dim=32 等情况导致报错
        curr_action_strong_full = pred_action_dict["normalized_action"]
        curr_action_strong = curr_action_strong_full[:, :, :7]
        
        N_samples = curr_action_strong.shape[0]
        full_action_horizon = curr_action_strong.shape[1]

        # 初始化变量
        candidate_ids = None      # 最终参与 Forward 对比的候选索引
        action_subset_strong = None # 对应的动作数据
        action_subset_weak = None   # 对应的弱策略数据
        dist_avg_prior = 0.0      # Backward Loss
        ratio = 0.0               # 平衡系数

        # === 逻辑分支: 判断是否进行 Backward 筛选 ===
        # 如果是第一帧(not last)，或者上一帧动作已经执行完了(无重叠)，则跳过 Backward
        need_backward = False
        if last_action_dict:
            last_chunk_size = last_action_dict["chunk_size"]
            if last_chunk_size < full_action_horizon:
                need_backward = True

        # ==========================================
        # 1. Backward Step (Backward Coherence)
        # ==========================================
        if not need_backward:
            # [Case A: t=0 或 无重叠] -> 直接使用所有样本做 Forward
            candidate_ids = np.arange(N_samples)
            action_subset_strong = curr_action_strong
            
            if pred_action_dict_weak is not None:
                action_subset_weak = pred_action_dict_weak["normalized_action"][:, :, :7]
            
            # ratio 保持为 0.0，即完全依赖 Forward Loss
            
        else:
            # [Case B: t>0 有重叠] -> 执行 Backward 筛选
            
            # --- 1.1 提取 Last Action Overlap ---
            last_act = last_action_dict["normalized_action"]
            if last_act.ndim == 2: last_act = last_act[None, ...] # [1, T, D]
            
            last_chunk_size = last_action_dict["chunk_size"]
            # 截取上一帧剩余的部分，并只取前7维
            last_action_overlap = last_act[:, last_chunk_size:, :7]
            overlap_len = last_action_overlap.shape[1]
            
            # --- 1.2 提取 Current Action Overlap ---
            # 截取当前预测的前 overlap_len 步
            # [Fix]: 使用 min 确保长度对齐，防止 index out of bounds
            final_len = min(overlap_len, curr_action_strong.shape[1])
            
            last_action_overlap = last_action_overlap[:, :final_len, :]
            curr_action_overlap = curr_action_strong[:, :final_len, :]
            
            # --- 1.3 执行 Backward 筛选 ---
            num_mode = 3
            num_sample_filter = max(1, N_samples // num_mode)
            
            # 筛选 Strong Policy
            # candidate_ids: 筛选后的索引 (在 0~19 之间)
            # dist_avg_prior: 对应的 Backward Loss
            candidate_ids, dist_avg_prior = backward(curr_action_overlap, last_action_overlap, num_sample=num_sample_filter, beta=beta)
            
            # 根据筛选出的 ID 提取数据用于 Forward
            action_subset_strong = curr_action_strong[candidate_ids]
            
            # 如果有 Weak Policy，也进行同样的 Overlap 提取和 Backward 筛选
            # (注意：Weak Policy 也要选出最连贯的作为负样本，而不是随机选)
            if pred_action_dict_weak is not None:
                curr_weak = pred_action_dict_weak["normalized_action"][:, :, :7]
                curr_weak_overlap = curr_weak[:, :final_len, :]
                
                # Weak 的筛选索引可能不同，我们要的是那组“最好的弱样本”
                weak_ids, _ = backward(curr_weak_overlap, last_action_overlap, num_sample=num_sample_filter, beta=beta)
                action_subset_weak = curr_weak[weak_ids]

            # --- 1.4 计算动态权重 Ratio ---
            # 只有在有 Backward 约束时才计算 Ratio
            PH = full_action_horizon
            AH = pred_chunk_size
            ratio = (PH * beta) ** 2 / ((PH * beta) ** 2 + AH ** 2)


        # ==========================================
        # 2. Forward Step (Forward Contrast)
        # ==========================================
        # 此时 action_subset_strong 的形状是 [K, T, 7]
        # K = N_samples (Case A) 或 num_sample_filter (Case B)
        
        # 2.1 准备广播数据
        # src: [K, 1, T, D]
        # tar: [1, K, T, D]
        src = action_subset_strong[:, None, :, :]
        tar = action_subset_strong[None, :, :, :]
        
        # 2.2 计算成对距离矩阵 [K, K]
        dist_pos_mat = euclidean_distance(src, tar, reduction='mean')
        
        # 2.3 计算 Positive Score (内聚性)
        num_candidates = dist_pos_mat.shape[0]
        top_k = num_candidates // 2 + 1
        
        # 对每行进行排序 (从小到大)
        sorted_indices = np.argsort(dist_pos_mat, axis=-1)
        # 取最近的 top_k 个邻居
        closest_indices = sorted_indices[:, :top_k]
        # 取出对应的距离值
        closest_values = np.take_along_axis(dist_pos_mat, closest_indices, axis=-1)
        # 去掉第一个（自己到自己，距离为0），取剩余邻居的平均值
        dist_avg_pos = closest_values[:, 1:].mean(axis=-1)
        
        # 2.4 计算 Negative Score (对比性)
        dist_avg_neg = 0.0
        if action_subset_weak is not None:
            # Strong [K, 1, T, D] vs Weak [1, K_weak, T, D]
            # 注意: Case B 下 K 和 K_weak 大小通常一致，但代码应兼容不一致的情况
            src = action_subset_strong[:, None, :, :]
            tar = action_subset_weak[None, :, :, :]
            
            dist_neg_mat = euclidean_distance(src, tar, reduction='mean') # [K, K_weak]
            
            # 同样取最近的 top_k 个弱样本 (为了计算它们有多近，然后远离它们)
            top_k_neg = dist_neg_mat.shape[1] // 2
            sorted_indices_neg = np.argsort(dist_neg_mat, axis=-1)
            closest_indices_neg = sorted_indices_neg[:, :top_k_neg]
            closest_values_neg = np.take_along_axis(dist_neg_mat, closest_indices_neg, axis=-1)
            
            dist_avg_neg = closest_values_neg.mean(axis=-1)

        # ==========================================
        # 3. Final Selection
        # ==========================================
        # Total Loss = Backward_Loss * ratio + (Pos_Score - Neg_Score) * (1 - ratio)
        # 解释: 
        # - dist_avg_prior 越小越好 (连贯)
        # - dist_avg_pos 越小越好 (合群)
        # - dist_avg_neg 越小越差 (我们希望离弱样本远，即距离大)，所以 Loss 里是减去它
        
        total_loss = dist_avg_prior * ratio + (dist_avg_pos - dist_avg_neg) * (1 - ratio)
        
        # 在候选子集中找到 Loss 最小的索引 (best_sub_idx)
        best_sub_idx = np.argmin(total_loss)
        
        # 映射回原始全量索引
        best_idx = candidate_ids[best_sub_idx]
        
        return best_idx

    else:
        raise ValueError(f"Unknown method: {method}")