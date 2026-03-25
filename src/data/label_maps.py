
"""
Utilities for creating, storing, and loading label mappings for multi-task
classification problems.

This module provides the LabelMaps class, which manages the number of classes
per task and optional id-to-label mappings, with support for construction from
a pandas DataFrame and JSON serialization.
"""


import json
import pandas as pd
from pathlib import Path
from typing import Dict, List


class LabelMaps:
    """
    Container for label metadata in multi-task classification.

    Stores the number of classes for each task and optional mappings from
    class indices to label names. Provides helper methods to construct from
    a DataFrame and to save/load mappings from disk.
    """

    def __init__(
        self,
        num_classes: Dict[str, int],
        id2label: Dict[str, Dict[int, str]] | None = None,
    ):
        self.num_classes = num_classes          
        self.id2label = id2label or {}    

    @classmethod
    def from_df(cls, df: pd.DataFrame, label_cols: List[str]) -> "LabelMaps":
        num_classes: Dict[str, int] = {}
        id2label: Dict[str, Dict[int, str]] = {}

        for col in label_cols:
            task_name = col.replace("label_", "")
            
            n = int(df[col].max() + 1)
            num_classes[task_name] = n

           
            id2label[task_name] = {i: str(i) for i in range(n)}

        return cls(num_classes=num_classes, id2label=id2label)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_classes": self.num_classes,
            "id2label": self.id2label,
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "LabelMaps":
        with open(path) as f:
            payload = json.load(f)
        return cls(
            num_classes=payload["num_classes"],
            id2label=payload.get("id2label", {}),
        )