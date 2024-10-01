#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import imageio


def generate_gif(data_dir: str, save_dir: str):
    with imageio.get_writer(
        Path(save_dir) / "intensity.gif", mode="I", loop=0, fps=10
    ) as writer:
        for file in sorted(Path(data_dir).glob("*.npz")):
            img = np.load(file, allow_pickle=True)["arr_0"].item()["intensity_img"]
            writer.append_data(img)


if __name__ == "__main__":
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--data_dir",
            type=str,
            required=True,
            help="Path to the directory containing point clouds",
        )
        parser.add_argument(
            "--save_dir", type=str, required=True, help="Path to save the results"
        )

        return parser.parse_args()

    args = parse_args()
    generate_gif(args.data_dir, args.save_dir)
