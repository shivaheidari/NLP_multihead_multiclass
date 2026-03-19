

"""
    Holds num_classes and optional id2label mappings for each task.
    Task names are derived from label column names by stripping 'label_'.
    Assumes labels in df are already integer-encoded starting from 0.
    """




import json
import pandas as pd
from pathlib import Path
from typing import Dict, List


class LabelMaps:
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