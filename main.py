#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing point clouds",
    )
    parser.add_argument("--nonground_dir", type=str, help="Path to nonground points")
    parser.add_argument(
        "-gn",
        "--gen_nonground",
        action="store_true",
        help="Generate nonground points",
    )
    parser.add_argument("--ego_motion", type=str, help="Path to ego motion file")

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    nonground_dir = (
        Path(args.nonground_dir)
        if args.nonground_dir is not None
        else data_dir / (data_dir.stem + "_xyz")
    )

    if args.gen_nonground or not nonground_dir.exists():
        from ground_removal import GroundRemoval

        ground_removal = GroundRemoval(args.data_dir)
        ground_removal.run()

    if args.ego_motion is None or args.gen_nonground:
        os.system(f"kiss_icp_pipeline {nonground_dir}")


if __name__ == "__main__":
    args = parse_args()

    main(args)
