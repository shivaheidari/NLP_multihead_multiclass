import torch
from transformers import AutoModel

from src.models.biobert_multitask import BioBertMultiHead


def test_biobert_multihead_forward_no_labels():
    encoder = AutoModel.from_pretrained("bert-base-uncased")
    num_classes = {
        "modality": 3,
        "vendor": 4,
        "series_type": 5,
        "plane": 2,
        "acquisition": 3,
        "body": 4,
        "contrast": 2,
    }
    model = BioBertMultiHead(encoder=encoder, num_classes_dict=num_classes)

    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs["loss"] is None
    assert outputs["logits_modality"].shape == (batch_size, num_classes["modality"])
    assert outputs["logits_vendor"].shape == (batch_size, num_classes["vendor"])


def test_biobert_multihead_forward_with_labels():
    encoder = AutoModel.from_pretrained("bert-base-uncased")
    num_classes = {
        "modality": 3,
        "vendor": 4,
        "series_type": 5,
        "plane": 2,
        "acquisition": 3,
        "body": 4,
        "contrast": 2,
    }
    model = BioBertMultiHead(encoder=encoder, num_classes_dict=num_classes)

    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    labels = {
        name: torch.randint(0, n, (batch_size,))
        for name, n in num_classes.items()
    }

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels_modality=labels["modality"],
        labels_vendor=labels["vendor"],
        labels_series_type=labels["series_type"],
        labels_plane=labels["plane"],
        labels_acquisition=labels["acquisition"],
        labels_body=labels["body"],
        labels_contrast=labels["contrast"],
    )

    assert outputs["loss"] is not None
    assert outputs["loss"].dim() == 0  # scalar
