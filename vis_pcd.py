#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        dest="pcd_path",
        type=str,
        required=True,
        help="Path to the point cloud file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="scala3",
        help="Name of the dataset, default: scala3",
    )
    parser.add_argument(
        "-p", "--processed", action="store_true", help="Data file is processed"
    )
    parser.add_argument(
        "-i", "--index", type=int, help="Index of the point cloud file, if applicable"
    )

    return parser.parse_args()


class Visualizer:
    def __init__(self, args):
        self.args = args

    def load_pcd(self, pcd_path: str) -> np.ndarray:
        if self.args.dataset == "scala3":
            if self.args.processed:
                pcd = np.load(pcd_path, allow_pickle=True)["raw_points"][
                    self.args.index
                ]
            else:
                pcd = np.load(pcd_path, allow_pickle=True)["arr_0"].item()["pc"][:, :3]
        elif self.args.dataset == "pone":
            if self.args.processed:
                raise NotImplementedError("Processed PONE dataset is not supported yet")
            else:
                scan = np.load(pcd_path, allow_pickle=True)["scan_list"][
                    self.args.index
                ]
                pcd = np.concatenate([scan["x"], scan["z"].reshape(-1, 1)], axis=1)
        elif self.args.dataset == "rm_ground":
            pcd = np.loadtxt(pcd_path)
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
        return pcd

    def visualize(self, xyz: np.ndarray) -> None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        o3d.visualization.draw_geometries([pcd])

    def run(self):
        if Path(self.args.pcd_path).is_dir():
            files = sorted(Path(self.args.pcd_path).glob("*.*"))
            if self.args.index is not None:
                xyz = self.load_pcd(files[self.args.index])
                self.visualize(xyz)
            else:
                for file in files:
                    xyz = self.load_pcd(file)
                    self.visualize(xyz)
        else:
            xyz = self.load_pcd(self.args.pcd_path)
            self.visualize(xyz)


if __name__ == "__main__":
    args = parse_args()

    vis = Visualizer(args)
    vis.run()
