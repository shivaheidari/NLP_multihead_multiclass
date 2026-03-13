# scripts/eval.py
import argparse
import json
from pathlib import Path

import torch
import yaml
import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import DicomDataModule  
from src.models import MultiHeadModel 
from src.metrics.classification import MultiHeadMetrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt/.ckpt.")
    return parser.parse_args()


def load_config(cfg_path: str):
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_label_info(processed_dir: Path):
    with open(processed_dir / "label_info.json") as f:
        return json.load(f)


def build_model(cfg, label_info):
    num_classes = label_info["num_classes"]          # dict: head -> int
    id2label = label_info["id2label"]                # optional, for logging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiHeadModel(
        backbone_name=cfg["model"]["backbone_name"],
        num_classes=num_classes,
        id2label=id2label,                           # if your model uses it
    )
    model.to(device)
    return model, device


def build_test_loader(cfg):
    dm = DicomDataModule(cfg)
    dm.setup("test")
    return dm.test_dataloader()


def evaluate(model, device, test_loader, heads):
    model.eval()
    metrics = MultiHeadMetrics(heads=heads)
    metrics.reset()

    with torch.no_grad():
        for batch in test_loader:
            # adjust to your batch structure
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_dict = {h: batch[f"label_{h}"].to(device) for h in heads}

            logits_dict = model(input_ids=input_ids, attention_mask=attention_mask)

            metrics.update_batch(logits_dict, labels_dict)

    return metrics.compute()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    processed_dir = Path(cfg["data"]["processed_dir"])
    label_info = load_label_info(processed_dir)
    heads = list(label_info["num_classes"].keys())

    model, device = build_model(cfg, label_info)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)

    test_loader = build_test_loader(cfg)
    results = evaluate(model, device, test_loader, heads)

    print("Test metrics per head:")
    for head, m in results.items():
        print(f"Head: {head}")
        for k, v in m.items():
            if k == "confusion_matrix":
                print(f"  {k}:")
                for row in v:
                    print("   ", row)
            else:
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
