import argparse
from pathlib import Path
import shutil

import yaml

from presp.evaluator import BuildingEvaluator
from presp.evolution import Evolution
from presp.prescriptor import BuildingPrescriptorFactory


def main():
    """
    Main logic for running neuroevolution.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, nargs="+", required=True)
    args = parser.parse_args()
    for config_path in args.config:
        print(f"Running evolution with config: {config_path}")
        with open(Path(config_path), "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if Path(config["save_path"]).exists():
                inp = input(f"Save path {config['save_path']} already exists. Do you want to overwrite? [Y|n].")
                if inp.lower() != "y":
                    print("Exiting.")
                    break
                shutil.rmtree(Path(config["save_path"]))
            prescriptor_factory = BuildingPrescriptorFactory(**config.pop("prescriptor_params"))
            evaluator = BuildingEvaluator()
            evolution = Evolution(prescriptor_factory=prescriptor_factory, evaluator=evaluator, **config)
            evolution.run_evolution()


if __name__ == "__main__":
    main()