
import torch

class DicomCollator:
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