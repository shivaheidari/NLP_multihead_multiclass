# NLP Multi-Head Multi-Class Classification for DICOM Metadata

This project implements a multi-head, multi-class Natural Language Processing (NLP) model to classify and normalize DICOM image text descriptions into structured metadata. It leverages a pre-trained transformer encoder (BioBERT) to simultaneously predict 7 different metadata attributes from a single input text.

## Features

- **Multi-Head Architecture:** Utilizes a single BioBERT text encoder with multiple separate linear classification heads.
- **Target Attributes:** Simultaneously predicts 7 distinct DICOM classes:
  - `modality`
  - `vendor`
  - `series_type`
  - `plane`
  - `acquisition`
  - `body`
  - `contrast`
- **Imbalanced Data Handling:** Supports applying specific class weights for imbalanced heads (`body` and `plane`/`contrast`).
- **Comprehensive Metrics:** Tracks accuracy, F1-macro, F1-weighted, precision, recall, and outputs a confusion matrix per classification head using `scikit-learn`.
- **Easy Inference:** Provides a plug-and-play `MultiHeadPredictor` to run inference on new, raw text strings.

---

## Project Structure

```text
NLP_multihead_multiclass/
├── scripts/
│   └── build_dataset.py               # Script to split data, encode labels, and prep for training
├── src/
│   ├── data/
│   │   └── dataset.py                 # PyTorch Dataset (DicomText) for loading inputs and 7 labels
│   ├── metadata_normalizer/
│   │   └── inference/
│   │       ├── predictor.py           # MultiHeadPredictor for performing text inference
│   │       └── config_loader.py       # Configuration loading utilities
│   ├── metrics/
│   │   └── classification.py          # MultiHeadMetrics for evaluation per head
│   ├── models/
│   │   └── biobert_multitask.py       # Core PyTorch model class (BioBertMultiHead)
│   └── prep/
│       └── preprocessing.py           # TagEncoder for transforming `<harmonized>` string data to IDs
└── README.md
```

---

## Usage

### 1. Data Preprocessing

To prepare the dataset, run the `build_dataset.py` script. This reads your raw data, splits it into Train/Validation/Test sets, fits the `TagEncoder` (only on the training set to prevent data leakage), and saves the processed CSVs alongside a `label_info.json` mapping.

```bash
python scripts/build_dataset.py --config configs/config.yaml
```

*Note: Your `configs/config.yaml` should define `raw_data.file`, `raw_data.test_size`, `raw_data.val_size`, and `data.processed_dir`.*

### 2. Training
*(Assuming you have a training loop setup integrating `BioBertMultiHead`, `DicomText`, and `MultiHeadMetrics`)*

During training, the `BioBertMultiHead` module natively calculates a combined Cross-Entropy loss for all 7 heads. You can optionally apply class weights for heavily imbalanced classes.

```python
from src.models.biobert_multitask import BioBertMultiHead
from transformers import AutoModel

encoder = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
model = BioBertMultiHead(encoder=encoder, num_classes_dict=num_classes)

# Optional: handle imbalanced classes
model.set_head_class_weights(plane_weights=my_plane_tensor, body_weights=my_body_tensor)
```

### 3. Inference

You can easily use the trained model to predict DICOM parameters on raw text using the `MultiHeadPredictor`.

```python
from src.metadata_normalizer.inference.predictor import MultiHeadPredictor

# Initialize the predictor with your compiled model weights and label mappings
predictor = MultiHeadPredictor(
    model_dir="outputs/models/biobert_multitask",
    label_info_path="data/processed"
)

# Run inference on new text
texts = [
    "Sagittal T1-weighted brain with contrast",
    "Axial CT Chest without IV contrast"
]
predictions = predictor.predict(texts)

for i, text in enumerate(texts):
    print(f"Input: {text}")
    print(f"Predictions: {predictions[i]}\n")
```

---

## Dependencies

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- pandas
- pyyaml

```bash
pip install torch transformers scikit-learn pandas pyyaml
```