#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Dict
from pathlib import Path

import tqdm
import numpy as np

from ground_removal import GroundRemoval


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the directory containing point clouds",
    )
    parser.add_argument("--data_nonground", type=str, help="Path to nonground points")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=5,
        help="Number of frames to be considered, default: 5",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="scala3",
        help="Name of the dataset, default: scala3",
    )

    return parser.parse_args()


class Dataset_creator:
    def __init__(self, args):
        self.args = args
        self.ground_removal = GroundRemoval(Path(args.data_root) / args.dataset)

    def run(self):
        file_num = len(list((Path(args.data_root) / args.dataset).glob("*.npz")))
        file_num = np.round(file_num / args.num_frames).astype(int) * args.num_frames

        for idx in tqdm.tqdm(range(0, file_num, args.num_frames)):
            data = self._create_dataframe(idx)

            file_name = (
                Path(args.data_root).resolve()
                / f"{args.dataset}_processed"
                / f"{args.dataset}_data_{idx}.npz"
            )
            file_name.parent.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(
                file_name,
                **data,
            )

    def _load_pcd(self, file_name: str) -> np.ndarray:
        path = Path(self.args.data_root) / self.args.dataset / (file_name + ".npz")
        pcd = np.load(path, allow_pickle=True)["arr_0"].item()["pc"][:, :3]
        return pcd

    def _load_nonground(self, file_name: str) -> np.ndarray:
        if self.args.data_nonground is None:
            nonground = self.ground_removal.run_individual(file_name + ".npz")
        else:
            path = Path(self.args.data_nonground) / (file_name + "_nonground.xyz")
            nonground = np.loadtxt(path)
        return nonground

    def _load_ego_pose(self, path: Path, idx: int) -> np.ndarray:
        pose = np.load(path, allow_pickle=True)[idx]
        return pose

    def _create_dataframe(self, start_idx: int) -> Dict:
        data = {
            "raw_points": np.array([]),
            "time_indice": np.array([]),
            "nonground": np.array([]),
            "ego_poses": np.array([]),
        }
        for i in range(args.num_frames):
            # Get the file name
            file_name = sorted(
                (Path(self.args.data_root) / self.args.dataset).glob("*.npz")
            )[start_idx + i].stem

            # Load data
            pcd = self._load_pcd(file_name)
            nonground = self._load_nonground(file_name)
            ego_pose = (  # later replace with ego motion from odometry
                self._load_ego_pose(
                    Path(__file__).parent.resolve()
                    / "results"
                    / "latest"
                    / f"{args.dataset}_xyz_poses.npy",
                    start_idx + i,
                ).reshape(1, 4, 4),
            )

            # Store data
            if i == 0:
                data["raw_points"] = pcd
                data["time_indice"] = np.full(pcd.shape[0], i)
                data["nonground"] = nonground
                data["ego_poses"] = ego_pose
            else:
                data["raw_points"] = np.concatenate((data["raw_points"], pcd))
                data["time_indice"] = np.concatenate(
                    (data["time_indice"], np.full(pcd.shape[0], i))
                )
                data["nonground"] = np.concatenate((data["nonground"], nonground))
                data["ego_poses"] = np.concatenate((data["ego_poses"], ego_pose))
        return data


if __name__ == "__main__":
    args = parse_args()

    creator = Dataset_creator(args)
    creator.run()
