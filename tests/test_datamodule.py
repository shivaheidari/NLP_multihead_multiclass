import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import DicomDataModule
import pandas as pd
from transformers import AutoTokenizer
from src.data import DicomDataModule



def test_dicom_data_module():

    cfg = {"data":
           {
               "train_csv": "data/processed/train.csv", 
                "val_csv": "data/processed/val.csv", 
                "batch_size": 1, 
                "max_length": 8

                   }}
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # faster than BioBERT for test 
    dicom = DicomDataModule(cfg, tokenizer)

    dicom.setup()

    train_loader = dicom.train_dataloader()
    batch = next(iter(train_loader))

    assert "input_ids" in batch
    assert "labels_modality" in batch
    assert "labels_body" in batch
    assert batch["input_ids"].shape[0] == 1