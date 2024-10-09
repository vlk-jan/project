#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

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
        "-s", "--save_dir", type=str, help="Path to save the nonground points"
    )
    parser.add_argument(
        "-p", "--processed", action="store_true", help="Data file is processed"
    )
    parser.add_argument(
        "-i", "--index", type=int, help="Index of the point cloud file, if applicable"
    )

    return parser.parse_args()


def load_pcd(
    pcd_path: str, dataset: str, idx: int, processed: bool = False
) -> np.ndarray:
    if dataset == "scala3":
        if processed:
            pcd = np.load(pcd_path, allow_pickle=True)["raw_points"][idx]
        else:
            pcd = np.load(pcd_path, allow_pickle=True)["arr_0"].item()["pc"][:, :3]
    elif dataset == "pone":
        if processed:
            raise NotImplementedError("Processed PONE dataset is not supported yet")
        else:
            scan = np.load(pcd_path, allow_pickle=True)["scan_list"][idx]
            pcd = np.concatenate([scan["x"], scan["z"].reshape(-1, 1)], axis=1)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return pcd


def visualize(xyz: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    args = parse_args()

    xyz = load_pcd(args.pcd_path, args.dataset, args.index, args.processed)
    visualize(xyz)
