import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModel

from src.data import DicomDataModule
from src.models.biobert_multitask import BioBertMultiHead
from src.training.trainer import fit


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    datamodule = DicomDataModule(cfg, tokenizer)

    encoder = AutoModel.from_pretrained(cfg["model"]["encoder_name"])
    model = BioBertMultiHead(encoder=encoder, num_classes_dict=cfg["model"]["num_classes"])
    model.to(device)

    fit(cfg, model, datamodule, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
