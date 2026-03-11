from torch.utils.data import Dataset

class DicomText(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df[[
            "label_modality",
            "label_vendor",
            "label_series_type",
            "label_plane",
            "label_acquisition",
            "label_body",
            "label_contrast",
        ]].to_dict(orient="list")
       

    def __getitem__(self, idx):
        text = self.texts[idx]
        item = {"text": text}
        item["label_modality"]   = self.labels["label_modality"][idx]
        item["label_vendor"]     = self.labels["label_vendor"][idx]
        item["label_series_type"]= self.labels["label_series_type"][idx]
        item["label_plane"]      = self.labels["label_plane"][idx]
        item["label_acquisition"]= self.labels["label_acquisition"][idx]
        item["label_body"]       = self.labels["label_body"][idx]
        item["label_contrast"]   = self.labels["label_contrast"][idx]
        return item

    def __len__(self):
        return len(self.texts)
