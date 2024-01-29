import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-angle", action="store_true")
    parser.add_argument("--joint-velocity", action="store_true")
    parser.add_argument("--position", action="store_true")
    parser.add_argument("--linear-velocity", action="store_true")
    parser.add_argument("--root", action="store_true")
    parser.add_argument("--number-of-controllable-joints", type=int, default=17)
    parser.add_argument("--number-of-bodies", type=int, default=18)

    arg = parser.parse_args()
    
    n_joints = arg.number_of_controllable_joints
    n_bodies = arg.number_of_bodies

    joint_angle_idx = [[4 * k, 4 * k + 1, 4 * k + 2] for k in range(n_joints)]

    offset = n_joints * 4
    linear_velocity_idx = [[offset + 9 * k, offset + 9 * k + 1, offset + 9 * k + 2]
                           for k in range(n_joints)]
    angular_velocity_idx = [[offset + 9 * k + 3, offset + 9 * k + 4, offset + 9 * k + 5]
                            for k in range(n_joints)]
    position_idx = [[offset + 9 * k + 6, offset + 9 * k + 7, offset + 9 * k + 8]
                    for k in range(n_joints)]

    offset = 13 * n_joints + n_bodies
    root_idx = [[3 * k + offset, 3 * k + offset + 1, 3 * k + offset + 2] for k in range(3)]

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
