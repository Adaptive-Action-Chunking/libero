import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def find_file_with_keywords(folder, keywords):
    matched = []
    for root, _, files in os.walk(folder):
        for file in files:
            if any(kw in file for kw in keywords):
                matched.append(os.path.join(root, file))
    return matched


target_dir = "/home/sangfor/Videos/libero_output/action_entropy_v2_h16_0922_gaussian_bernoulli_0/libero_goal_20250922_154616"
# target_dir = "/home/sangfor/Videos/libero_output/action_entropy_v2_h16_gaussian_bernoulli_0/libero_goal_20250920_225947"
# target_dir = "/home/sangfor/Videos/libero_output/action_entropy_v2_h16_gaussian_bernoulli_0/libero_10_20250920_174101"
# target_dir = "/home/sangfor/Videos/libero_output/action_entropy_v2_h16_gaussian_bernoulli_0/libero_spatial_20250920_214742"
# chunk mean
# target_dir = "/home/sangfor/Videos/libero_output/action_entropy_v2_h16_chunk_mean_gaussian_bernoulli_0/libero_10_20250923_171705"



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


sns.set(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(data=merged, x="chunk_size", discrete=True, stat="probability", color="gray")
# sns.histplot(data=merged, x="chunk_size", bins=8, stat="probability", color="gray")
# sns.countplot(data=merged, x="chunk_size", stat="probability")
plt.title("Chunk Size Distribution (All Episodes)")
plt.xlabel("Chunk size")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# chunk size for success 
plt.figure(figsize=(8, 5))
sns.histplot(data=merged[merged["success"] == 1], x="chunk_size", discrete=True, hue="success", multiple="dodge", stat="probability")
plt.title("Chunk Size Distribution by Success")
plt.xlabel("Chunk size")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# chunk size for failure
plt.figure(figsize=(8, 5))
sns.histplot(data=merged[merged["success"] == 0], x="chunk_size", discrete=True, hue="success", multiple="dodge", stat="probability")
plt.title("Chunk Size Distribution by Failure")
plt.xlabel("Chunk size")
plt.ylabel("Count")
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))
sns.histplot(data=merged, x="chunk_size", discrete=True, hue="success", multiple="dodge", stat="probability")
plt.title("Chunk Size Distribution by Success/Failure")
plt.xlabel("Chunk size")
plt.ylabel("Count")
plt.legend(title="Success", labels = ["Failure (0)", "Success (1)"])
plt.tight_layout()
plt.show()

# g = sns.FacetGrid(merged, col="task_id", hue="success", col_wrap=5, sharex=False, sharey=False)
# g.map(sns.histplot, "chunk_size", discrete=True, stat="probability", multiple="dodge")
# g.add_legend(title="Success")
# plt.subplots_adjust(top=0.9)
# g.fig.suptitle("Chunk Size Distribution per Task")
# plt.show()

# # chunk size
# chunk_success = merged[merged["success"] == 1].groupby("task_id")["chunk_size"].describe()
# chunk_fail = merged[merged["success"] == 0].groupby("task_id")["chunk_size"].describe()
# chunk_all = merged.groupby("task_id")["chunk_size"].describe()

# # 推理次数

# inference_counts = merged.groupby(["task_id", "trial_id"])["inference_id"].nunique().reset_index(name="num_inferences")
# inference_counts = inference_counts.merge(results[["task_id", "trial_id", "success"]],
#                                           on=["task_id", "trial_id"], how="left"
#                                           )

# inference_success = inference_counts[inference_counts["success"] == 1].groupby("task_id")["num_inferences"].describe()
# inference_fail = inference_counts[inference_counts["success"] == 0].groupby("task_id")["num_inferences"].describe()
# inference_all = inference_counts.groupby("task_id")["num_inferences"].describe()


