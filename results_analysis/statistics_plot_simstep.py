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


# target_dir = "/home/sangfor/Videos/libero_output/action_entropy_v2_h16_0922_gaussian_bernoulli_0/libero_object_20250920_201930"
target_dir = "/home/sangfor/Videos/libero_output/action_entropy_v2_h16_0922_gaussian_bernoulli_0/libero_goal_20250922_154616"
# target_dir = "/home/sangfor/Videos/libero_output/action_entropy_v2_h16_0922_gaussian_bernoulli_0/libero_10_20250920_174101"
# target_dir = "/home/sangfor/Videos/libero_output/action_entropy_v2_h16_0922_gaussian_bernoulli_0/libero_spatial_20250920_214742"

# chunk mean
# target_dir = "/home/sangfor/Videos/libero_output/action_entropy_v2_h16_chunk_mean_gaussian_bernoulli_0/libero_10_20250923_171705"

results_file = find_file_with_keywords(target_dir, ["results_log"])[0]
actions_file =  find_file_with_keywords(target_dir, ["chunk_size_log"])[0]

results = pd.read_csv(results_file)
actions = pd.read_csv(actions_file)

merged = actions.merge(
    results[["task_id", "trial_id", "success"]],
    on=["task_id", "trial_id"],
    how="left"
    )

task_id_select = 1
task_data = merged[merged["task_id"] == task_id_select]

# data_to_use = task_data
data_to_use = merged

# 热力图全部一起
mean_curve = (
    data_to_use.groupby("sim_step", as_index=False)["chunk_size"]
    .mean()
    .rename(columns={"chunk_size":"chunk_size_mean"})
)

# 滑动平均
window_size = 10
mean_curve["chunk_size_smooth"] = (
    mean_curve["chunk_size_mean"].rolling(window = window_size, center=True).mean()
)


plt.figure(figsize=(10, 6))
sns.histplot(
    data=data_to_use,
    x="sim_step",
    y="chunk_size",
    bins=15,
    pmax=0.9,
    cbar=True,
    cmap="viridis"
)

plt.plot(
    mean_curve["sim_step"],
    mean_curve["chunk_size_smooth"],
    color="red",
    linewidth=2,
    label="Mean Chunk size"
)

plt.xlabel("Simulation step")
plt.ylabel("Chunk size")
plt.tight_layout()
plt.show()

# 热力图分任务
# g= sns.FacetGrid(
#     merged,
#     col="task_id",
#     col_wrap=3,
#     sharex=True,
#     sharey=True,
#     height=4
# )

# g.map_dataframe(
#     sns.histplot,
#     x="sim_step",
#     y="chunk_size",
#     bins=15,
#     cmap="viridis",
#     cbar = False
# )

# g.set_axis_labels("Simulation_step", "Chunk Size")
# g.set_titles("Task: {col_name}")
# # plt.subplots_adjust()
# plt.show()

# # 箱形图
# merged["sim_step_bin"] = (merged["sim_step"] // 10) * 10
# g = sns.FacetGrid(merged, col="task_id", col_wrap=3, sharey=True, sharex=True, height=4)
# g.map_dataframe(sns.boxplot,
#                 x="sim_step_bin",
#                 y="chunk_size",
#                 order = sorted(merged["sim_step_bin"].unique())
#                 )
# g.set_axis_labels("Simulation Step", "Chunk Size")
# g.set_titles("Task: {col_name}")
# plt.subplots_adjust(top=0.9)
# # g.fig.suptitle("")
# plt.show()









