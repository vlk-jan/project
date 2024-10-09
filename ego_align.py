#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import numpy as np

from vis_pcd import load_pcd, visualize


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


def load_transform(
    pcd_path: str, dataset: str, idx: int, processed: bool = False
) -> np.ndarray:
    if dataset == "scala3":
        if processed:
            transform = np.load(pcd_path, allow_pickle=True)["ego_motion"][idx]
        else:
            raise ValueError("Unprocessed Scala3 dataset does not have ego motion data")
    elif dataset == "pone":
        if processed:
            raise NotImplementedError("Processed PONE dataset is not supported yet")
        else:
            transform = np.load(pcd_path, allow_pickle=True)["odom_list"][idx][
                "transformation"
            ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform points using the given transformation matrix

    :param np.ndarray points:
        Points to be transformed

    :param np.ndarray transform:
        Transformation matrix

    :return: np.ndarray:
        Transformed points
    """
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = np.dot(points, transform.T)
    return points[:, :3]


def align(args):
    pcd1 = load_pcd(args.pcd_path, args.dataset, args.index, args.processed)
    pcd2 = load_pcd(args.pcd_path, args.dataset, args.index + 1, args.processed)
    transform1 = load_transform(args.pcd_path, args.dataset, args.index, args.processed)
    transform2 = load_transform(
        args.pcd_path, args.dataset, args.index + 1, args.processed
    )
    pcd1_t = transform_points(pcd1, transform1)
    pcd2_t = transform_points(pcd2, transform2)

    pcd_vis = np.concatenate([pcd1_t, pcd2_t], axis=0)
    visualize(pcd_vis)


if __name__ == "__main__":
    args = parse_args()

    align(args)
