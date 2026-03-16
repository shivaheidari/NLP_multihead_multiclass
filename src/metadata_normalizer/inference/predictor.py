from pathlib import Path
from typing import List, Dict
import json
import torch
from transformers import AutoTokenizer, AutoModel

from src.models.biobert_multitask import BioBertMultiHead
from metadata_normalizer.inference.config_loader import load_project_config


class MultiHeadPredictor:
    """
    High-level inference wrapper.

    Usage:
        predictor = MultiHeadPredictor("outputs/models/biobert_multitask")
        preds = predictor.predict(["series description"])
    """

    def __init__(self, model_dir: str, device: str | None = None, max_length: int = 128):
        self.model_dir = Path(model_dir)
        self.max_length = max_length

        # 1. Load label info (num_classes, id2label)
        with open(self.model_dir / "label_info.json") as f:
            label_info = json.load(f)
        self.num_classes = label_info["num_classes"]   # dict[str, int]
        self.id2label = label_info["id2label"]         # dict[str, dict[str,int]] or similar

        # 2. Decide encoder name
        cfg  = load_project_config()
        encoder_name = cfg["model"]["encoder_name"]

        # 3. Setup device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 4. Load tokenizer and encoder
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        encoder = AutoModel.from_pretrained(encoder_name)

        # 5. Build model and load weights
        self.model = BioBertMultiHead(encoder=encoder, num_classes_dict=self.num_classes)
        self.model.to(self.device)
        self._load_checkpoint()

        self.model.eval()

    def _load_checkpoint(self):
        ckpt_path = self.model_dir / "model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)

    def predict(self, texts: List[str]) -> List[Dict[str, str]]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            logits = outputs["logits"] 

        preds: List[Dict[str, str]] = []
        batch_size = enc["input_ids"].size(0)

        for i in range(batch_size):
            head_labels: Dict[str, str] = {}
            for head, head_logits in logits.items():
                pred_id = torch.argmax(head_logits[i]).item()
                id2lab = self.id2label[head]
                if isinstance(next(iter(id2lab.keys())), str):
                    label_str = id2lab[str(pred_id)]
                else:
                    label_str = id2lab[pred_id]
                head_labels[head] = label_str
            preds.append(head_labels)

        return preds
