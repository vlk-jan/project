#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Dict
from pathlib import Path

import tqdm
import numpy as np


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
    parser.add_argument("--verbose", action="store_true", help="Print verbose messages")

    return parser.parse_args()


class Dataset_creator:
    def __init__(self, args):
        self.args = args

        self.file_list = sorted(
            (Path(self.args.data_root) / self.args.dataset).glob("*.npz")
        )

        if args.dataset == "scala3":
            pass
        elif args.dataset == "pone":
            self.pone_data = np.load(
                Path(args.data_root) / args.dataset / self.file_list[0],
                allow_pickle=True,
            )
            self.scan_data = self.pone_data["scan_list"]
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        if args.data_nonground is None:
            from ground_removal import GroundRemoval

            self.ground_removal = GroundRemoval(args, standalone=True)

        self.ego_poses = self._load_ego_poses()
        del self.pone_data

    def run(self):
        proc_num = (
            np.round(self.ego_poses.shape[0] / self.args.num_frames).astype(int)
            * self.args.num_frames
        )

        for idx in tqdm.tqdm(range(0, proc_num, self.args.num_frames)):
            data = self._create_dataframe(idx)

            file_name = (
                Path(self.args.data_root).resolve()
                / f"{self.args.dataset}_processed"
                / f"{self.args.dataset}_data_{idx // self.args.num_frames:03d}.npz"
            )
            file_name.parent.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(
                file_name,
                **data,
            )

    def _load_pcd(self, idx: int) -> np.ndarray:
        if self.args.dataset == "scala3":
            path = (
                Path(self.args.data_root)
                / self.args.dataset
                / (self.file_list[idx].stem + ".npz")
            )
            pcd = np.load(path, allow_pickle=True)["arr_0"].item()["pc"][:, :3]
        elif self.args.dataset == "pone":
            pcd = np.concatenate(
                [
                    self.scan_data[idx]["x"],
                    self.scan_data[idx]["z"].reshape(-1, 1),
                ],
                axis=1,
            )
        else:
            raise ValueError(f"Unknown dataset: {self.args}")

        return pcd

    def _load_nonground(self, idx: int) -> np.ndarray:
        if self.args.data_nonground is None:
            nonground = self.ground_removal.run_individual_scan(self._load_pcd(idx))[1]
        else:
            file_name = (
                self.file_list[idx].stem
                if self.args.dataset == "scala3"
                else f"{self.file_list[0].stem}_{idx:04d}"
            )
            path = Path(self.args.data_nonground) / (file_name + "_nonground.xyz")
            nonground = np.loadtxt(path)[:, -1]
        return nonground.astype(int)

    def _load_ego_poses(self) -> np.ndarray:
        if self.args.dataset == "scala3":
            path = (
                Path(__file__).parent.resolve()
                / "results"
                / "latest"
                / f"{args.dataset}_xyz_poses.npy"
            )
            poses = np.load(path, allow_pickle=True)
        elif self.args.dataset == "pone":
            poses = self.pone_data["odom_list"]
        else:
            raise ValueError(f"Unknown dataset: {self.args}")
        return poses

    def _create_dataframe(self, start_idx: int) -> Dict:
        data = {
            "raw_points": np.array([]),
            "time_indice": np.array([]),
            "nonground": list(),
            "ego_poses": np.array([]),
        }

        for i in range(args.num_frames):
            # Load data
            pcd = self._load_pcd(start_idx + i)
            nonground = self._load_nonground(start_idx + i)
            ego_pose = (
                self.ego_poses[start_idx + i].reshape(1, 4, 4)
                if self.args.dataset == "scala3"
                else self.ego_poses[start_idx + i]["transformation"].reshape(1, 4, 4)
            )

            # Store data
            if i == 0:
                data["raw_points"] = pcd
                data["time_indice"] = np.full(pcd.shape[0], i)
                data["nonground"].append(nonground)
                data["ego_poses"] = ego_pose
            else:
                data["raw_points"] = np.concatenate((data["raw_points"], pcd))
                data["time_indice"] = np.concatenate(
                    (data["time_indice"], np.full(pcd.shape[0], i))
                )
                data["nonground"].append(nonground)
                data["ego_poses"] = np.concatenate((data["ego_poses"], ego_pose))
        data["nonground"] = np.array(data["nonground"], dtype=object)

        return data


if __name__ == "__main__":
    args = parse_args()

    creator = Dataset_creator(args)
    creator.run()
