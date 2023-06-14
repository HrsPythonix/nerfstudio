import json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

plt.switch_backend("agg")

vis_flag = True
vis_image_path = "test.jpg"
cluster_num = 3
filename = "/cfs/risheng/workspace/NS_test/postbank_data/postbank_counter_clip1_a7s3_4k_800/transforms.json"
output_path = os.path.join(os.path.dirname(filename), "transforms_clustered.json")
with open(filename, encoding="UTF-8") as file:
    meta = json.load(file)

frames_num = len(meta["frames"])
meta["frames"] = sorted(meta["frames"], key=lambda frame: frame["file_path"])

pose_origins = []
for i in range(frames_num):
    pose = np.array(meta["frames"][i]["transform_matrix"])
    origin = pose[:3, 3]
    pose_origins.append(origin)

pose_origins = np.array(pose_origins)

pose_clips = KMeans(n_clusters=cluster_num, random_state=2023).fit_predict(pose_origins)

if vis_flag:
    ax = plt.axes(projection="3d")
    ax.scatter(pose_origins[:, 0], pose_origins[:, 1], pose_origins[:, 2], c=pose_clips)
    plt.show()
    plt.savefig(vis_image_path)

for i, pose_clip in enumerate(pose_clips):
    meta["frames"][i]["clip"] = int(pose_clip)

with open(output_path, "w", encoding="UTF-8") as f:
    json.dump(meta, f, indent=4)
