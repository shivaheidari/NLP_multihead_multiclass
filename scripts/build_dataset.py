'''

to run this:
python scripts/build_dataset.py --config configs/config.yaml
'''
import argparse
import yaml
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prep.preprocessing import TagEncoder


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    raw_cfg = cfg["raw_data"]
    data_cfg = cfg["data"]

    
    raw_df = pd.read_csv(raw_cfg["file"])
    test_size = raw_cfg.get("test_size", 0.1)
    val_size  = raw_cfg.get("val_size", 0.1)

    train_val_df, test_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=42,
        shuffle=True,
        stratify=None, 
    )

    # adjust val_size relative to train_val
    val_size_rel = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_rel,
        random_state=42,
        shuffle=True,
        stratify=None,
    )

    # 3. Fit TagEncoder on TRAIN ONLY
    encoder = TagEncoder()
    encoder.fit(train_df)
    num_classes = encoder.num_classes  # dict head -> int

    out_dir = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "label_info.json", "w") as f:
        json.dump(
            {
                "label2id": encoder.label2id,
                "id2label": encoder.id2label,
                "num_classes": num_classes,
            },
            f,
        )

    # 4. Transform all splits
    train_proc = encoder.transform(train_df)
    val_proc   = encoder.transform(val_df)
    test_proc  = encoder.transform(test_df)

    # 5. Save processed CSVs
    out_dir = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_proc.to_csv(out_dir / "train.csv", index=False)
    val_proc.to_csv(out_dir / "val.csv", index=False)
    test_proc.to_csv(out_dir / "test.csv", index=False)

    # 6. Save label mappings for inference
    maps_path = out_dir / "label_maps.json"
    with open(maps_path, "w") as f:
        json.dump(
            {"label2id": encoder.label2id, "id2label": encoder.id2label},
            f,
        )

    print("Saved processed datasets to:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
