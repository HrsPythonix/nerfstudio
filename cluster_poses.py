import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances, silhouette_score


def se3_distance(pose1, pose2):
    pose1 = pose1.reshape(4, 4)
    pose2 = pose2.reshape(4, 4)
    rotation1 = R.from_matrix(pose1[:3, :3])
    rotation2 = R.from_matrix(pose2[:3, :3])
    translation_distance = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
    rotation_distance = rotation1.inv() * rotation2
    rotation_angle = rotation_distance.magnitude()
    return translation_distance + rotation_angle


def se3_affinity(x):
    return pairwise_distances(x, metric=se3_distance)


def run_kmeans(pose_origins, cluster_num):
    best_cluster_num = cluster_num
    if cluster_num == -1:
        best_score = -1
        best_cluster_num = 2
        max_cluster_num = 30
        for cluster_num in range(2, max_cluster_num + 1):
            kmeans = KMeans(n_clusters=cluster_num, random_state=2023, n_init=20, max_iter=500)
            cluster_labels = kmeans.fit_predict(pose_origins)
            silhouette_avg = silhouette_score(pose_origins, cluster_labels)
            print("For n_clusters =", cluster_num, "The average silhouette_score is :", silhouette_avg)

            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_cluster_num = cluster_num
    print("Best n_clusters =", best_cluster_num)

    pose_clips = KMeans(n_clusters=best_cluster_num, random_state=2023, n_init=20, max_iter=500).fit_predict(
        pose_origins
    )
    return pose_clips


def run_dbscan(poses, pose_origins, use_se3):
    if use_se3:
        clustering = DBSCAN(eps=0.5, min_samples=5, metric=se3_distance)
        pose_clips = clustering.fit_predict(poses.reshape(poses.shape[0], -1))
    else:
        clustering = DBSCAN(eps=0.5, min_samples=5)
        pose_clips = clustering.fit_predict(pose_origins)
    return pose_clips


def run_AC(poses, pose_origins, cluster_num, use_se3):
    if use_se3:
        clustering = AgglomerativeClustering(n_clusters=cluster_num, affinity=se3_affinity, linkage="complete")
        pose_clips = clustering.fit_predict(poses.reshape(poses.shape[0], -1))
    else:
        clustering = AgglomerativeClustering(n_clusters=cluster_num)
        pose_clips = clustering.fit_predict(pose_origins)
    return pose_clips


# 添加新的聚类方法
def run_SC(poses, pose_origins, cluster_num, use_se3):
    if use_se3:
        pose_clips = SpectralClustering(
            n_clusters=cluster_num, assign_labels="kmeans", random_state=2023, affinity=se3_affinity
        ).fit_predict(poses.reshape(poses.shape[0], -1))
    else:
        pose_clips = SpectralClustering(n_clusters=cluster_num, assign_labels="kmeans", random_state=2023).fit_predict(
            pose_origins
        )
    return pose_clips


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="相机姿态聚类")

    parser.add_argument("--vis", action="store_true", help="进行可视化")
    parser.add_argument("--server", action="store_true", help="服务器运行")

    parser.add_argument("--vis_image_path", type=str, default="test.jpg", help="可视化图片保存路径")
    parser.add_argument("--cluster_num", type=int, default=-1, help="聚类数量")
    parser.add_argument("--random_seed", type=int, default=2023, help="随机种子")
    parser.add_argument(
        "--input_path",
        type=str,
        default="D:/workspace/NeRFRealData/17F_Test/20230607_dining_ip12_hloc_600/transforms.json",
        help="输入文件路径",
    )
    parser.add_argument("--output_path", type=str, required=True, help="输出文件路径")

    # KMeans, AC, DBSCAN, SC
    parser.add_argument("--cluster_method", type=str, default="KMeans", help="聚类算法")
    parser.add_argument("--use_se3", action="store_true", help="使用se3距离度量, 不兼容KMeans")

    parser.set_defaults(vis=False, server=False, use_se3=False)
    args = parser.parse_args()

    if args.server:
        plt.switch_backend("agg")

    with open(args.input_path, encoding="UTF-8") as file:
        meta = json.load(file)

    frames_num = len(meta["frames"])
    meta["frames"] = sorted(meta["frames"], key=lambda frame: frame["file_path"])

    poses = []
    pose_origins = []
    for i in range(frames_num):
        pose = np.array(meta["frames"][i]["transform_matrix"])
        origin = pose[:3, 3]
        poses.append(pose)
        pose_origins.append(origin)
    poses = np.array(poses)
    pose_origins = np.array(pose_origins)

    # distance_matrix = np.zeros((frames_num, frames_num))
    # for i in range(frames_num):
    #     for j in range(frames_num):
    #         distance_matrix[i, j] = se3_distance(poses[i], poses[j])

    pose_clips = None
    if args.cluster_method == "KMeans":
        pose_clips = run_kmeans(pose_origins, args.cluster_num)
    elif args.cluster_method == "DBSCAN":
        pose_clips = run_dbscan(poses, pose_origins, args.use_se3)
    elif args.cluster_method == "AC":
        pose_clips = run_AC(poses, pose_origins, args.cluster_num, args.use_se3)
    elif args.cluster_method == "SC":
        pose_clips = run_SC(poses, pose_origins, args.cluster_num, args.use_se3)

    if pose_clips is not None:
        if args.vis:
            ax = plt.axes(projection="3d")
            ax.scatter(pose_origins[:, 0], pose_origins[:, 1], pose_origins[:, 2], c=pose_clips)
            plt.show()
            plt.savefig(args.vis_image_path)
        print(pose_clips)

        for i, pose_clip in enumerate(pose_clips):
            meta["frames"][i]["clip"] = [f"clip_{int(pose_clip)}"]

        with open(args.output_path, "w", encoding="UTF-8") as f:
            json.dump(meta, f, indent=4)
