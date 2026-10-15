"""Train FeRF on scene archives. Run from the repository root."""

import argparse
from ferf.engine import train
from ferf.utils import load_config, resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int, help="Override total optimizer steps")
    parser.add_argument("--data-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.steps is not None: config["train"]["steps"] = args.steps
    if args.data_root: config["data"]["root"] = args.data_root
    if args.output: config["train"]["output"] = args.output
    checkpoint = train(config, resolve_device(args.device), args.resume)
    print(f"Checkpoint saved: {checkpoint}")


if __name__ == "__main__":
    main()
