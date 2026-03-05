import json
import pandas as pd
from pathlib import Path

LABEL_COLS = [
    "label_modality",
    "label_vendor",
    "label_series_type",
    "label_plane",
    "label_acquisition",
    "label_body",
    "label_contrast",
]

def build_label_maps_from_df(df: pd.DataFrame):
    id2label = {}
    label2id = {}
    num_classes = {}

    for col in LABEL_COLS:
        values = sorted(df[col].unique())
        mapping = {int(v): int(v) for v in values}  # if already ints, trivial
        # or if you had strings, build real maps here
        num_classes[col.replace("label_", "")] = len(values)
        # optionally store label2id/id2label if labels are strings
    return num_classes

def save_num_classes(num_classes: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(num_classes, f)

def load_num_classes(path: str):
    with open(path) as f:
        return json.load(f)
