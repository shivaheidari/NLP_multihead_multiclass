from src.data.dataset import DicomText
import pandas as pd

import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_dicom_text_basic():
    df = pd.DataFrame(

        {
            "text": ["a", "b"],
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
    

    assert len(ds) == 2

    sample = ds[0]

    assert "text" in sample
    assert sample["text"] == "a"
    assert sample["label_modality"] == 0
    assert sample["label_body"] == 2