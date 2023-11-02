import numpy as np
import os
import torch
import argparse
from matplotlib import pyplot as plt

from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-count", type=int, default=51)
    parser.add_argument("--obs-count", type=int, default=252)

    arg = parser.parse_args()

    action_count = arg.action_count
    obs_count = arg.obs_count

    joint_angle_idx = [[4 * k, 4 * k + 1, 4 * k + 2] for k in range(17)]
    linear_velocity_idx = [[68 + 9 * k, 68 + 9 * k + 1, 68 + 9 * k + 2] for k in range(17)]
    angular_velocity_idx = [[68 + 9 * k + 3, 68 + 9 * k + 4, 68 + 9 * k + 5] for k in range(17)]
    position_idx = [[17 * 4 + 9 * k + 6, 17 * 4 + 9 * k + 7, 17 * 4 + 9 * k + 8] for k in range(17)]
    root_idx = [k + 239 for k in range(9)]

    idx_list = linear_velocity_idx
    for idx in idx_list:
        print('{')
        print(f"\t\"start\": {idx[0]},")
        print(f"\t\"end\": {idx[2]}")
        print('},')

    print(root_idx)
