import argparse
from pathlib import Path

from yukarin_autoreg.config import create_from_json
from yukarin_autoreg.trainer import create_trainer


def train(
        config_json_path: Path,
        output: Path,
):
    config = create_from_json(config_json_path)
    trainer = create_trainer(config=config, output=output)
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_json_path', type=Path)
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()

    train(
        config_json_path=arguments.config_json_path,
        output=arguments.output,
    )
