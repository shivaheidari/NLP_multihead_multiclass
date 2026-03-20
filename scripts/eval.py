"""
Helper script to evaluate the trained multi-head multi-class classification model.

What it is:
    An evaluation script that tests a saved model checkpoint against the test dataset.

What it does:
    1. Loads the project configuration and label mappings.
    2. Initializes the BioBertMultiHead model and loads the specified checkpoint weights.
    3. Prepares the test dataloader using the DicomDataModule.
    4. Runs inference on the test set and calculates metrics (Accuracy, F1, 
       Precision, Recall, and Confusion Matrix) for all 7 classification heads.
    5. Prints the results to the console and saves them to text files in the outputs directory.

How to run:
    python scripts/eval.py --config <path_to_config.yaml> --checkpoint <path_to_model_checkpoint>

Example:
    python scripts/eval.py --config configs/config.yaml --checkpoint outputs/models/biobert_multitask/finetunedweighted.bin
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import DicomDataModule  
from src.models.biobert_multitask import BioBertMultiHead 
from src.metrics.classification import MultiHeadMetrics
from transformers import AutoModel, AutoTokenizer

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

    model = BioBertMultiHead(
        encoder=AutoModel.from_pretrained(cfg["model"]["encoder_name"]),
        num_classes_dict=num_classes
    )
    model.to(device)
    return model, device


def build_test_loader(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    dm = DicomDataModule(cfg, tokenizer=tokenizer)
    dm.setup()
    return dm.test_dataloader()


def evaluate(model, device, test_loader, heads):
    model.eval()
    metrics = MultiHeadMetrics(heads=heads)
    metrics.reset()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits_dict = {
                "modality":    outputs["logits_modality"],
                "vendor":      outputs["logits_vendor"],
                "series_type": outputs["logits_series_type"],
                "plane":       outputs["logits_plane"],
                "acquisition": outputs["logits_acquisition"],
                "body":        outputs["logits_body"],
                "contrast":    outputs["logits_contrast"],
            }

            labels_dict = {
                "modality":    batch["labels_modality"].to(device),
                "vendor":      batch["labels_vendor"].to(device),
                "series_type": batch["labels_series_type"].to(device),
                "plane":       batch["labels_plane"].to(device),
                "acquisition": batch["labels_acquisition"].to(device),
                "body":        batch["labels_body"].to(device),
                "contrast":    batch["labels_contrast"].to(device),
            }

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

    for head, m in results.items():


        with open(f"outputs/{head}.txt", "w+") as f:
            for k, v in m.items():
                if k == "confusion_matrix":
                    f.write(f"{k}\n")
                    for row in v:
                        f.write(f"{row}\n")
                else:
                    f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
