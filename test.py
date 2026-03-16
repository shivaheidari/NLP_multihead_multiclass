from metadata_normalizer import MultiHeadPredictor
predictor = MultiHeadPredictor(model_dir="outputs/models/biobert_multitask")
texts = [
    "Sagittal T1-weighted brain with contrast",
    "Axial FLAIR brain without contrast",
]
preds = predictor.predict(texts)
print(preds)