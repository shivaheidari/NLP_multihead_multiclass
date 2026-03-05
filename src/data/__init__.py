import pandas as pd
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .dataset import DicomText
from .collate import DicomCollator


def build_dataloaders(cfg: dict, tokenizer:PreTrainedTokenizerBase):
    
    data_cfg = cfg["data"]
    train_df = pd.read_csv(data_cfg["train_csv"])
    val_df = pd.read_csv(data_cfg["val_csv"])


    train_dataset = DicomText(train_df)
    val_dataset = DicomText(val_dataset)


    collator = DicomCollator(
        tokenizer=tokenizer,
         max_length=cfg["max_length"], )
    

    train_loader = DataLoader(

        train_dataset, 
        batch_size=data_cfg["batch_size"], 
        shuffle = True,
        collate_fn=collator,
    )

    val_loader = DataLoader(

        train_dataset, 
        batch_size=data_cfg["batch_size"], 
        shuffle = True,
        collate_fn=collator,
    )