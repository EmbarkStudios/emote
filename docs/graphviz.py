"""Take a directory, and process all .dot files in it.

For each .dot file, generate a .png file into the output directory.
"""

import argparse
import os

from dataclasses import dataclass


@dataclass
class Args:
    input_dir: str
    output_dir: str

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Process .dot files in a directory into .png files"
        )
        parser.add_argument("input_dir", type=str, help="Input directory")
        parser.add_argument("output_dir", type=str, help="Output directory")
        args = parser.parse_args()

        return Args(args.input_dir, args.output_dir)


def main(input_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if not file.endswith(".dot"):
            continue

        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file.replace(".dot", ".png"))
        print(f"Processing {input_file} into {output_file}")
        os.system(f"dot -Tpng {input_file} -o {output_file}")


if __name__ == "__main__":
    args = Args.parse_args()
    main(args.input_dir, args.output_dir)
