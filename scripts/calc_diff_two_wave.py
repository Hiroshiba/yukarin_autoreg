import argparse
from pathlib import Path

from yukarin_autoreg.evaluator import calc_mcd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path1', type=Path)
    parser.add_argument('path2', type=Path)
    arguments = parser.parse_args()

    diff = calc_mcd(
        path1=arguments.path1,
        path2=arguments.path2,
    )
    print(diff)
