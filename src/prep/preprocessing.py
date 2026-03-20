"""
    This module provides:
    label2id / id2label mappings from a 'harmonized' column and
    encodes each row into numeric labels per head.
"""

import re
from typing import Dict, Set

import pandas as pd


class TagEncoder:
    """
    TagEncoder class for preprocessing the multihead-multiclass calssification 
    provides label2id and id2label per each head. 
    List of heads are 
    modality,
    vendor,
    series_type,
    plane,
    acquisition,
    body,
    contrast

    Args:
        None

    Attributes:
        HEADS (List[str]): List of classification heads.

    Methods:
    fit: accepts dataframe obj
    transform: creates preprocessed dataframe

    Usage:
        # 1. Initialize the TagEncoder
        encoder = TagEncoder()
        
        #2. Fit on a DataFrame
            encoder.fit(df)
        
        #3. Transform a DataFrame
        df = encoder.transform(df)
    
    """
    
    HEADS = ["modality", "vendor", "series_type", "plane", "acquisition", "body", "contrast"]
    

    def __init__(self):
        self.label2id: Dict[str, Dict[str, int]] = {}
        self.id2label: Dict[str, Dict[int, str]] = {}

    @staticmethod
    def _parse_harmonized(harmonized: str) -> Dict[str, str]:
       
        tokens = re.findall(r"<([^>]+)>", harmonized or "")
        expected_slots = len(TagEncoder.HEADS)
        if len(tokens) < expected_slots:
            tokens = tokens + ["none"] * (expected_slots - len(tokens))

        return {
            "modality":    tokens[0],
            "vendor":      tokens[1],
            "series_type": tokens[2],
            "plane":       tokens[3],
            "acquisition": tokens[4],
            "body":        tokens[5],
            "contrast":    tokens[6],
        }
  

    def fit(self, df: pd.DataFrame) -> None:

        labels_per_head: Dict[str, Set[str]] = {h: set() for h in self.HEADS}

        for harmonized in df["harmonized"]:
            slots = self._parse_harmonized(harmonized)
            for head, value in slots.items():
                labels_per_head[head].add(value)

        # ensures 'none' exists for all heads
        for head in labels_per_head:
            labels_per_head[head].add("none")

        # builds mappings
        for head, values in labels_per_head.items():
            sorted_values = sorted(values)
            l2i = {v: i for i, v in enumerate(sorted_values)}
            i2l = {i: v for v, i in l2i.items()}
            self.label2id[head] = l2i
            self.id2label[head] = i2l

    @property
    def num_classes(self):
        return {head: len(self.label2id[head]) for head in self.HEADS}

    def encode_row(self, harmonized: str) -> Dict[str, int]:
       
        slots = self._parse_harmonized(harmonized)
        return {
            head: self.label2id[head][slots.get(head, "none")]
            for head in self.HEADS
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
      
        df = df.copy()

        encoded = df["harmonized"].apply(self.encode_row)

        for head in self.HEADS:
            df[f"label_{head}"] = encoded.apply(lambda d: d[head])

        return df
