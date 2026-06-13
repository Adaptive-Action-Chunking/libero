import pandas as pd
import os

def find_file_with_keywords(folder, keywords):
    matched = []
    for root, _, files in os.walk(folder):
        for file in files:
            if any(kw in file for kw in keywords):
                matched.append(os.path.join(root, file))
    return matched

# target_dir = "/home/sangfor/Videos/libero_output/action_entropy_v2_h16_0922_gaussian_bernoulli_0/libero_goal_20250922_154616"
target_dir = "/home/sangfor/Videos/libero_output/action_entropy_v2_h16_0922_gaussian_bernoulli_0/libero_10_20250922_104914"

# target_dir = "/home/sangfor/Videos/libero_output/fixed_size_baseline_0928/_fixed_8_0/libero_10_20250929_074618"
# target_dir = "/home/sangfor/Videos/libero_output/fixed_size_baseline_0928/_fixed_16_0/libero_10_20250928_220425"

results_file = find_file_with_keywords(target_dir, ["results_log"])[0]
actions_file =  find_file_with_keywords(target_dir, ["chunk_size_log"])[0]

results = pd.read_csv(results_file)
actions = pd.read_csv(actions_file)

merged = actions.merge(
    results[["task_id", "trial_id", "success"]],
    on=["task_id", "trial_id"],
    how="left"
    )

chunk_success = merged[merged["success"] == 1].groupby("task_id")["chunk_size"].describe()

chunk_fail = merged[merged["success"] == 0].groupby("task_id")["chunk_size"].describe()

chunk_all = merged.groupby("task_id")["chunk_size"].describe()

# 推理次数

inference_counts = merged.groupby(["task_id", "trial_id"])["inference_id"].nunique().reset_index(name="num_inferences")

inference_counts = inference_counts.merge(results[["task_id", "trial_id", "success"]],
                                          on=["task_id", "trial_id"], how="left"
                                          )

inference_success = inference_counts[inference_counts["success"] == 1].groupby("task_id")["num_inferences"].describe()

inference_fail = inference_counts[inference_counts["success"] == 0].groupby("task_id")["num_inferences"].describe()


inference_all = inference_counts.groupby("task_id")["num_inferences"].describe()

print("success episode chunk size:")
print(chunk_success)

print("failed episode chunk size:")
print(chunk_fail)

print("all episode chunk size:")
print(chunk_all)

print("success episode inference counts:")
print(inference_success)

print("failed episode inference counts:")
print(inference_fail)

print("all episode inference counts:")
print(inference_all)
