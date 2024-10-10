import os
import numpy as np
from utils_helper import transform_points
from utils_cluster import cluster_pcd


class Dataset_pca:
    def __init__(self, args):
        self.args = args
        self.seq_paths = os.listdir(self.args.root + self.args.dataset + "_processed")
        print(f"number of test sequences: {len(self.seq_paths)}")

    def load_data_pca(self, data_path):
        """
        Inut:
            raw_points:     [m, 3]              points before ego-motion compensation
            time_indice:    [m]
            ego_motion_gt:  [n_frames, 4, 4]
        """
        data = np.load(data_path, allow_pickle=True)
        raw_points, time_indice = data["raw_points"], data["time_indice"]
        ego_motion_gt, nonground = data["ego_poses"], data["nonground"]

        assert raw_points.shape[0] == time_indice.shape[0]
        assert ego_motion_gt.shape[0] == len(np.unique(time_indice))
        assert len(np.unique(time_indice)) == max(time_indice) + 1
        assert max(time_indice) + 1 == self.args.num_frames

        data = {
            "raw_points": raw_points,
            "time_indice": time_indice,
            "ego_motion_gt": ego_motion_gt,
            "nonground": nonground,
            "data_path": data_path,
        }

        return data

    def cluster_labels_two(self, data):
        points_src = []
        points_dst = []
        labels_src = []
        labels_dst = []
        for j in range(1, self.args.num_frames):
            # print(f'calculate scence flow between {j} and {0}')
            point_dst = data["raw_points"][data["time_indice"] == 0, 0:3]
            point_src = data["raw_points"][data["time_indice"] == j, 0:3]
            pose = data["ego_poses"][j]
            point_src_ego = transform_points(point_src, pose)
            points_tmp = np.concatenate([point_dst, point_src_ego], axis=0)

            nonground_dst = data["nonground"][data["time_indice"] == 0]
            nonground_src = data["nonground"][data["time_indice"] == j]
            nonground_tmp = np.concatenate([nonground_dst, nonground_src], axis=0)
            label_tmp = cluster_pcd(self.args, points_tmp, nonground_tmp)
            label_src = label_tmp[len(point_dst) :]
            label_dst = label_tmp[0 : len(point_dst)]

            labels_src.append(label_src)
            labels_dst.append(label_dst)
            points_src.append(point_src_ego)
            points_dst.append(point_dst)
        assert len(points_src) == len(points_dst)
        assert len(labels_src) == len(labels_dst)
        assert len(points_src) == len(labels_dst)
        return points_src, points_dst, labels_src, labels_dst

    def __len__(self):
        return len(self.seq_paths)

    def __getitem__(self, idx):
        data = self.load_data_pca(self.seq_paths[idx])

        points_src, points_dst, labels_src, labels_dst = self.cluster_labels_two(data)

        return data, points_src, points_dst, labels_src, labels_dst
