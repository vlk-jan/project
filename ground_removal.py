#!/bin/bash/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, Union, Optional

import numpy as np
import open3d as o3d
import pypatchworkpp


class GroundRemoval:
    def __init__(
        self,
        data_dir: Union[str, Path],
        save_dir: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        visualize: bool = False,
    ):
        params = pypatchworkpp.Parameters()
        params.verbose = verbose

        self.patchwork = pypatchworkpp.patchworkpp(params)
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.visualize = visualize

    def _read_pcd(self, pcd_path: Union[str, Path]) -> np.ndarray:
        pcd = np.load(pcd_path, allow_pickle=True)["arr_0"].item()
        pcd = pcd["pc"][:, :3]

        return pcd

    def _visualize(self):
        ground = self.patchwork.getGround()
        nonground = self.patchwork.getNonground()
        time_taken = self.patchwork.getTimeTaken()

        # Get centers and normals for patches
        centers = self.patchwork.getCenters()
        normals = self.patchwork.getNormals()

        print("Original Points  #: ", (ground.shape[0] + nonground.shape[0]))
        print("Ground Points    #: ", ground.shape[0])
        print("Nonground Points #: ", nonground.shape[0])
        print("Time Taken : ", time_taken / 1000000, "(sec)")
        print("Press ... \n")
        print("\t H  : help")
        print("\t N  : visualize the surface normals")
        print("\tESC : close the Open3D window")

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=600, height=400)

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

        ground_o3d = o3d.geometry.PointCloud()
        ground_o3d.points = o3d.utility.Vector3dVector(ground)
        ground_o3d.colors = o3d.utility.Vector3dVector(
            np.array(
                [[0.0, 1.0, 0.0] for _ in range(ground.shape[0])], dtype=float
            )  # RGB
        )

        nonground_o3d = o3d.geometry.PointCloud()
        nonground_o3d.points = o3d.utility.Vector3dVector(nonground)
        nonground_o3d.colors = o3d.utility.Vector3dVector(
            np.array(
                [[1.0, 0.0, 0.0] for _ in range(nonground.shape[0])], dtype=float
            )  # RGB
        )

        centers_o3d = o3d.geometry.PointCloud()
        centers_o3d.points = o3d.utility.Vector3dVector(centers)
        centers_o3d.normals = o3d.utility.Vector3dVector(normals)
        centers_o3d.colors = o3d.utility.Vector3dVector(
            np.array(
                [[1.0, 1.0, 0.0] for _ in range(centers.shape[0])], dtype=float
            )  # RGB
        )

        vis.add_geometry(mesh)
        vis.add_geometry(ground_o3d)
        vis.add_geometry(nonground_o3d)
        vis.add_geometry(centers_o3d)

        vis.run()
        vis.destroy_window()

    def _save_results(self, file_name: Union[str, Path]):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.save_dir / f"{file_name}_ground.npz",
            self.patchwork.getGround(),
        )
        np.savez_compressed(
            self.save_dir / f"{file_name}_nonground.npz",
            self.patchwork.getNonground(),
        )

    def run(self) -> Dict[str, np.ndarray]:
        nonground = {}
        for pcd_path in sorted(self.data_dir.glob("*.npz")):
            pcd = self._read_pcd(pcd_path)
            self.patchwork.estimateGround(pcd)

            nonground[pcd_path.stem] = self.patchwork.getNonground()

            if self.visualize:
                self._visualize()

            if self.save_dir is not None:
                self._save_results(pcd_path.stem)

        return nonground


if __name__ == "__main__":
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description="Ground Removal")

        parser.add_argument(
            "--data_dir",
            type=str,
            required=True,
            help="Path to the directory containing point clouds",
        )
        parser.add_argument("--save_dir", type=str, help="Path to save the results")
        parser.add_argument(
            "-d", "--debug", action="store_true", help="Print verbose messages"
        )
        parser.add_argument(
            "-v", "--visualize", action="store_true", help="Visualize the results"
        )

        return parser.parse_args()

    args = parse_args()
    ground_removal = GroundRemoval(
        args.data_dir, args.save_dir, args.debug, args.visualize
    )

    nonground = ground_removal.run()
