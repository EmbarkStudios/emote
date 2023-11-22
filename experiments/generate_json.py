import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-angle", action="store_true")
    parser.add_argument("--joint-velocity", action="store_true")
    parser.add_argument("--position", action="store_true")
    parser.add_argument("--linear-velocity", action="store_true")
    parser.add_argument("--root", action="store_true")

    arg = parser.parse_args()

    joint_angle_idx = [[4 * k, 4 * k + 1, 4 * k + 2] for k in range(17)]
    linear_velocity_idx = [[68 + 9 * k, 68 + 9 * k + 1, 68 + 9 * k + 2] for k in range(17)]
    angular_velocity_idx = [[68 + 9 * k + 3, 68 + 9 * k + 4, 68 + 9 * k + 5] for k in range(17)]
    position_idx = [[17 * 4 + 9 * k + 6, 17 * 4 + 9 * k + 7, 17 * 4 + 9 * k + 8] for k in range(17)]
    root_idx = [[3 * k + 239, 3 * k + 239 + 1, 3 * k + 239 + 2] for k in range(3)]

    list_of_indices = []
    if arg.joint_angle:
        list_of_indices.extend(joint_angle_idx)
    if arg.joint_velocity:
        list_of_indices.extend(angular_velocity_idx)
    if arg.position:
        list_of_indices.extend(position_idx)
    if arg.linear_velocity:
        list_of_indices.extend(linear_velocity_idx)
    if arg.root:
        list_of_indices.extend(root_idx)

    for idx in list_of_indices:
        print('{')
        print(f"\t\"start\": {idx[0]},")
        print(f"\t\"end\": {idx[2] + 1}")
        print('},')
