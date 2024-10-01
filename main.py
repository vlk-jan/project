#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ground_removal import GroundRemoval


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

        return parser.parse_args()

    args = parse_args()

    ground_removal = GroundRemoval(args.data_dir)
    nonground = ground_removal.run()
