# """
#     Handles loading CSVs, building DicomText datasets, and wrapping them
#     in DataLoaders with the DicomCollator.
#     Expects cfg['data'] to contain: train_csv, val_csv, batch_size, max_length.
#     """
# import pandas as pd
# from torch.utils.data import DataLoader
# from transformers import PreTrainedTokenizerBase

# from .dataset import DicomText
# from .collate import DicomCollator


# class DicomDataModule:
   

#     def __init__(self, cfg: dict, tokenizer: PreTrainedTokenizerBase):
#         data_cfg = cfg["data"]
#         self.train_csv = data_cfg["train_csv"]
#         self.val_csv   = data_cfg["val_csv"]
#         self.batch_size = data_cfg["batch_size"]
#         self.max_length = data_cfg["max_length"]
#         self.tokenizer = tokenizer

#         self.train_dataset = None
#         self.val_dataset = None
#         self.collator = None

#     def setup(self):
       
#         train_df = pd.read_csv(self.train_csv)
#         val_df   = pd.read_csv(self.val_csv)

        
#         self.train_dataset = DicomText(train_df)
#         self.val_dataset   = DicomText(val_df)

        
#         self.collator = DicomCollator(
#             tokenizer=self.tokenizer,
#             max_length=self.max_length,
#         )

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             collate_fn=self.collator,
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             collate_fn=self.collator,
#         )


import pandas as pd
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .dataset import DicomText
from .collate import DicomCollator


class DicomDataModule:
    """
    Handles loading CSVs, building DicomText datasets, and wrapping them
    in DataLoaders with the DicomCollator.
    Expects cfg['data'] to contain: train_csv, val_csv, test_csv, batch_size, max_length.
    """

    def __init__(self, cfg: dict, tokenizer: PreTrainedTokenizerBase):
        data_cfg = cfg["data"]
        self.train_csv = data_cfg["train_csv"]
        self.val_csv   = data_cfg["val_csv"]
        self.test_csv  = data_cfg["test_csv"]  # <-- new

        self.batch_size = data_cfg["batch_size"]
        self.max_length = data_cfg["max_length"]
        self.tokenizer  = tokenizer

        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None  
        self.collator      = None

    def setup(self, stage: str | None = None):
   
        if stage in (None, "fit"):
            train_df = pd.read_csv(self.train_csv)
            val_df   = pd.read_csv(self.val_csv)

            self.train_dataset = DicomText(train_df)
            self.val_dataset   = DicomText(val_df)

        if stage in (None, "test"):
            test_df = pd.read_csv(self.test_csv)
            self.test_dataset = DicomText(test_df)

        # collator is shared
        if self.collator is None:
            self.collator = DicomCollator(
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )
