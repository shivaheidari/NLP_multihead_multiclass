import torch
import pandas as pd
from transformers import AutoTokenizer

from src.data.dataset import DicomText
from src.data.collate import DicomCollator

import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_dicom_collator_shapes():
    df = pd.DataFrame(
        {
            "text": ["hello world", "another text"],
            "label_modality": [0, 1],
            "label_vendor": [1, 0],
            "label_series_type": [2, 3],
            "label_plane": [0, 1],
            "label_acquisition": [1, 1],
            "label_body": [2, 2],
            "label_contrast": [0, 0],
        }
    )
    ds = DicomText(df)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # faster than BioBERT for test
    collator = DicomCollator(tokenizer=tokenizer, max_length=8)

    batch = [ds[0], ds[1]]
    enc = collator(batch)

    assert "input_ids" in enc
    assert "attention_mask" in enc
    assert enc["input_ids"].shape == (2, 8)

    for key in [
        "labels_modality",
        "labels_vendor",
        "labels_series_type",
        "labels_plane",
        "labels_acquisition",
        "labels_body",
        "labels_contrast",
    ]:
        assert key in enc
        assert isinstance(enc[key], torch.Tensor)
        assert enc[key].shape == (2,)
