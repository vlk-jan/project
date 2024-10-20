#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import seaborn as sns

import argparse
from typing import Dict
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir", type=str, required=True, help="Path to the root directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="scala3",
        help="Name of the dataset, default: scala3",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=5,
        help="Number of frames to be considered, default: 5",
    )
    parser.add_argument(
        "--save", action="store_true", default=False, help="Save the flow data"
    )

    return parser.parse_args()


def visualize_pcd(
    points, labels=None, num_colors=3, title="visualization", if_save=False
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if labels is None:
        pass
    else:
        COLOR_MAP = sns.color_palette("husl", n_colors=num_colors)
        COLOR_MAP = np.array(COLOR_MAP)
        labels = labels.astype(int)
        colors = COLOR_MAP[labels % len(COLOR_MAP)]
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_flow(points, labels=None, title="visualization", if_save=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if labels is None:
        pass
    else:
        colors = labels / np.max(labels)
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"Flow: {title}")
    o3d.visualization.draw_geometries([pcd], window_name=title)


def load_data(args: argparse.Namespace, idx: int) -> Dict:
    file_path = (
        Path(args.root_dir)
        / (args.dataset + "_processed")
        / f"{args.dataset}_data_{idx:03d}.npz"
    )
    data = np.load(file_path, allow_pickle=True)
    return data


def load_flow(args: argparse.Namespace, idx: int) -> Dict:
    file_path = (
        Path(args.root_dir)
        / (args.dataset + "_flow")
        / f"{args.dataset}_flow_{idx:03d}.npz"
    )
    data = np.load(file_path, allow_pickle=True)
    return data["scene_flow"]


def save(point_src, point_dst, flow, idx):
    np.savez_compressed(
        f"flow_vis/flow_{idx}.npz",
        point_src=point_src,
        point_dst=point_dst,
        flow=flow,
    )


def main(args: argparse.Namespace):
    data_path = Path(args.root_dir) / (args.dataset + "_flow")
    for idx in range(len(list(data_path.glob("*.npz")))):
        data = load_data(args, idx)
        flow_data = load_flow(args, idx)

        point_dst = data["raw_points"][data["time_indice"] == 0]
        start = len(point_dst)

        for j in range(1, args.num_frames):
            point_src = data["raw_points"][data["time_indice"] == j]
            end = start + len(point_src)

            flow = flow_data[start:end]

            start += len(point_src)

            if args.save:
                save(point_src, point_dst, flow, f"{idx}_{j}")

            visualize_flow(point_src, flow, title=f"flow: {j} vs {0}")

            visualize_pcd(
                np.concatenate([point_src + flow, point_dst], axis=0),
                np.concatenate(
                    [np.zeros((len(point_src))) + 1, np.zeros((len(point_dst))) + 2],
                    axis=0,
                ),
                num_colors=3,
                title=f"temporal: src+flow vs dst: {j} vs {0}",
            )


if __name__ == "__main__":
    args = parse_args()

    main(args)
