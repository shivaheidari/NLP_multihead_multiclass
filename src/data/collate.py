"""
DicomCollator module is used for tokenizing, embedding, padding, and max_length settings. 
Attributes:
        tokenizer: A Hugging Face tokenizer used to encode input text.
        max_length (int): Maximum sequence length for tokenization. Sequences longer than
            this will be truncated, and shorter ones will be padded.

This module is callable for example: DicomCollator(texts, padding='max_size', max_length, return_tensors)

it returns tokenized text with the labels for each category including:
            label_modality,
            "label_vendor",
            "label_series_type",
            "label_plane",
            "label_acquisition",
            "label_body",
            "label_contrast",
"""

import torch

class DicomCollator:
    """
    Data collator for batching DICOM-related text samples with multiple classification labels.

    This collator tokenizes input text and aggregates multiple target labels into tensors,
    making it suitable for multi-head / multi-task classification models (e.g., modality,
    vendor, plane, etc.).

    Attributes:
        tokenizer: A Hugging Face tokenizer used to encode input text.
        max_length (int): Maximum sequence length for tokenization. Sequences longer than
            this will be truncated, and shorter ones will be padded.
    """
    
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [item["text"] for item in batch]

        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels_modality     = torch.tensor([b["label_modality"]     for b in batch], dtype=torch.long)
        labels_vendor       = torch.tensor([b["label_vendor"]       for b in batch], dtype=torch.long)
        labels_series_type  = torch.tensor([b["label_series_type"]  for b in batch], dtype=torch.long)
        labels_plane        = torch.tensor([b["label_plane"]        for b in batch], dtype=torch.long)
        labels_acquisition  = torch.tensor([b["label_acquisition"]  for b in batch], dtype=torch.long)
        labels_body         = torch.tensor([b["label_body"]         for b in batch], dtype=torch.long)
        labels_contrast     = torch.tensor([b["label_contrast"]     for b in batch], dtype=torch.long)

        enc["labels_modality"]    = labels_modality
        enc["labels_vendor"]      = labels_vendor
        enc["labels_series_type"] = labels_series_type
        enc["labels_plane"]       = labels_plane
        enc["labels_acquisition"] = labels_acquisition
        enc["labels_body"]        = labels_body
        enc["labels_contrast"]    = labels_contrast

        return enc