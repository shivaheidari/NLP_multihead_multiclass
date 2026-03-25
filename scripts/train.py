"""
Helper script to train the multi-head multi-class classification model.

What it is:
    A main training script that fine-tunes a pre-trained BioBERT model 
    to predict 7 DICOM metadata attributes simultaneously.

What it does:
    1. Loads the project configuration and label metadata.
    2. Initializes the tokenizer and prepares the DicomDataModule.
    3. Initializes the BioBertMultiHead model.
    4. Calculates and applies class weights for heavily imbalanced classes 
       (e.g., 'plane' and 'body') based on the training dataset.
    5. Executes the training loop (training and validation) and saves 
       the best model checkpoint to the output directory.

How to run:
    python scripts/train.py --config <path_to_config.yaml>

Example:
    python scripts/train.py --config configs/config.yaml
"""
import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
import pandas as pd

import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import DicomDataModule
from src.models.biobert_multitask import BioBertMultiHead
from src.training.trainer import fit
from src.training.utils import compute_class_weights

CUDA_LAUNCH_BLOCKING=1


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    datamodule = DicomDataModule(cfg, tokenizer)

    processed_dir = Path(cfg["data"]["processed_dir"])

    with open(processed_dir / "label_info.json") as f:
         label_info = json.load(f)
    num_classes = label_info["num_classes"]  # dict head -> int

    encoder = AutoModel.from_pretrained(cfg["model"]["encoder_name"])
    model = BioBertMultiHead(encoder=encoder, num_classes_dict=num_classes)
    model.to(device)
    
    train_df = pd.read_csv(cfg["data"]["train_csv"])

    plane_lables = train_df["label_plane"].tolist()
    body_labels = train_df["label_body"].tolist()

    plane_w = compute_class_weights(plane_lables, num_classes["plane"]).to(device)
    body_w = compute_class_weights(body_labels, num_classes["body"]).to(device)
    model.set_head_class_weights(plane_w, body_w)
    
    fit(cfg, model, datamodule, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a multi-head multi-class classification from prepared dataset with numerical class values", 
                                     epilog="Example: python scripts/train.py --config configs/config.yaml")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file containg raw_data and data setting", 
                        metavar="CONFIG_PATH")
    args = parser.parse_args()
    main(args.config)
