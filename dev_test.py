import sys, os
sys.path.insert(0, os.path.abspath("src")) 

from metadata_normalizer import MultiHeadPredictor

predictor = MultiHeadPredictor(
    model_dir="outputs/models/biobert_multitask",
    label_info_path="data/processed",  # or from your config
)

texts = [
    "Sagittal T1-weighted brain with contrast",
    "Axial FLAIR brain without contrast",
]
preds = predictor.predict(texts)
print(preds)
