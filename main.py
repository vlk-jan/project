#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import fileinput
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


def printinfo(args: argparse.Namespace) -> None:
    print("#################")
    print("# Configuration #")
    print("#################\n")
    print(f"\033[93mData directory: \033[96m{Path(args.data_dir).resolve()}\033[00m")
    if args.nonground_dir is not None:
        print(
            f"\033[93mNonground directory: \033[96m{Path(args.nonground_dir).resolve()}\033[00m"
        )
    else:
        print(
            f"\033[93mNonground directory: \033[96m{(Path(args.data_dir).parent / (Path(args.data_dir).stem + '_xyz')).resolve()}\033[00m"
        )
        print("    Nonground directory not provided, using default")
    print(
        f"\033[93mGenerate nonground: \033[{92 if args.gen_nonground else 91}m{args.gen_nonground}\033[00m"
    )
    print(
        "   If Nonground directory does not exist, nonground points will be generated"
    )
    if args.ego_motion is not None:
        print(f"\033[93mEgo motion: \033[96m{Path(args.ego_motion).resolve()}\033[00m")
    else:
        print("\033[93mEgo motion: \033[91mnot provided\033[00m")
        print("    Ego motion will be generated, saving to the directory:")
        print(
            f"    \033[96m{Path(__file__).resolve().parent / 'results' / 'latest' / ((Path(args.nonground_dir) if args.nonground_dir is not None else Path(args.data_dir).parent / (Path(args.data_dir).stem + '_xyz')).stem + '_poses.npy')}\033[00m"
        )
    print("#################\n")


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    nonground_dir = (
        Path(args.nonground_dir)
        if args.nonground_dir is not None
        else data_dir.parent / (data_dir.stem + "_xyz")
    )

    if args.gen_nonground or not nonground_dir.exists():
        from ground_removal import GroundRemoval

        print("Generating nonground points")
        ground_removal = GroundRemoval(args.data_dir)
        ground_removal.run()

    if args.ego_motion is None or args.gen_nonground:
        print("Generating ego motion")

        config_file = Path(__file__).parent / "config" / "kiss_icp.yml"
        os.system(f"cp {config_file.parent / 'kiss_icp_template.yml'} {config_file}")
        for line in fileinput.input(f"{config_file}", inplace=True):
            print(line.replace("<cwd>", str(Path(__file__).parent.resolve())), end="")

        os.system(f"kiss_icp_pipeline {nonground_dir} --config {config_file}")


if __name__ == "__main__":
    args = parse_args()

    printinfo(args)

    main(args)
