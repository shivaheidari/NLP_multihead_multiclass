
from typing import Dict, Any

import os
import torch
from torch.utils.data import DataLoader


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create AdamW optimizer with different LRs for encoder and heads."""
    training_cfg = cfg["training"]

    encoder_lr = float(training_cfg["encoder_lr"])
    head_lr = float(training_cfg["head_lr"])
    weight_decay = float(training_cfg.get("weight_decay", 0.0))

    # split params: encoder vs heads
    encoder_params = list(model.encoder.parameters())
    head_params = []

    head_params += list(model.modality_head.parameters())
    head_params += list(model.vendor_head.parameters())
    head_params += list(model.series_type_head.parameters())
    head_params += list(model.plane_head.parameters())
    head_params += list(model.acq_head.parameters())
    head_params += list(model.body_head.parameters())
    head_params += list(model.contrast_head.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params,    "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,) -> float:
    
    
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Very simple evaluation: per-head accuracy + mean accuracy."""
    model.eval()

    correct = {
        "modality": 0,
        "vendor": 0,
        "series_type": 0,
        "plane": 0,
        "acquisition": 0,
        "body": 0,
        "contrast": 0,
    }
    total = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        # predictions
        pred_modality    = outputs["logits_modality"].argmax(dim=1)
        pred_vendor      = outputs["logits_vendor"].argmax(dim=1)
        pred_series_type = outputs["logits_series_type"].argmax(dim=1)
        pred_plane       = outputs["logits_plane"].argmax(dim=1)
        pred_acquisition = outputs["logits_acquisition"].argmax(dim=1)
        pred_body        = outputs["logits_body"].argmax(dim=1)
        pred_contrast    = outputs["logits_contrast"].argmax(dim=1)

        # labels
        labels_modality    = batch["labels_modality"]
        labels_vendor      = batch["labels_vendor"]
        labels_series_type = batch["labels_series_type"]
        labels_plane       = batch["labels_plane"]
        labels_acquisition = batch["labels_acquisition"]
        labels_body        = batch["labels_body"]
        labels_contrast    = batch["labels_contrast"]

        batch_size = labels_modality.size(0)
        total += batch_size

        correct["modality"]    += (pred_modality    == labels_modality).sum().item()
        correct["vendor"]      += (pred_vendor      == labels_vendor).sum().item()
        correct["series_type"] += (pred_series_type == labels_series_type).sum().item()
        correct["plane"]       += (pred_plane       == labels_plane).sum().item()
        correct["acquisition"] += (pred_acquisition == labels_acquisition).sum().item()
        correct["body"]        += (pred_body        == labels_body).sum().item()
        correct["contrast"]    += (pred_contrast    == labels_contrast).sum().item()

    acc = {name + "_acc": correct[name] / total for name in correct}
    acc["mean_accuracy"] = sum(acc.values()) / len(correct)
    return acc


def fit(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    datamodule,
    device: torch.device,
) -> None:
    """
    Full training loop:
    - builds train/val dataloaders from datamodule
    - builds optimizer
    - runs epochs with train + eval
    - saves best model state_dict
    """
    training_cfg = cfg["training"]

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader   = datamodule.val_dataloader()

    optimizer = build_optimizer(cfg, model)

    num_epochs = training_cfg["epochs"]
    best_val = None
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        val_primary = val_metrics["mean_accuracy"]

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_mean_acc={val_primary:.4f}")

        if best_val is None or val_primary > best_val:
            best_val = val_primary
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        output_dir = cfg["paths"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        torch.save(best_state_dict, os.path.join(output_dir, cfg["training"]["model_name"]))
