import argparse
import json
import torch
from ferf.checkpoint import model_from_checkpoint
from ferf.data import SceneDataset
from ferf.engine import evaluate
from ferf.rendering import Renderer
from ferf.utils import resolve_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate all non-source views of held-out scenes")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--source-view", type=int, default=0)
    parser.add_argument("--output", default="runs/eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--limit", type=int, help="Debug subset; omit for full evaluation")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1: parser.error("--limit must be positive")
    torch.set_num_threads(4)
    device = resolve_device(args.device)
    model, state = model_from_checkpoint(args.checkpoint, device)
    data_config = dict(state["config"]["data"])
    if args.data_root: data_config["root"] = args.data_root
    dataset = SceneDataset(**data_config, split=args.split, source_view=args.source_view, evaluation=True)
    result = evaluate(model, Renderer(**state["config"]["render"]), dataset, device, args.output, args.lpips, args.limit)
    print(json.dumps(result["mean_over_views"], indent=2))


if __name__ == "__main__":
    main()
