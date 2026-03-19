from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel


class BioBertMultiHead(nn.Module):
    def __init__(self, encoder: PreTrainedModel, num_classes_dict: Dict[str, int]):
        super().__init__()
        self.encoder = encoder
        hidden_size = self.encoder.config.hidden_size

        self.modality_head    = nn.Linear(hidden_size, num_classes_dict["modality"])
        self.vendor_head      = nn.Linear(hidden_size, num_classes_dict["vendor"])
        self.series_type_head = nn.Linear(hidden_size, num_classes_dict["series_type"])
        self.plane_head       = nn.Linear(hidden_size, num_classes_dict["plane"])
        self.acq_head         = nn.Linear(hidden_size, num_classes_dict["acquisition"])
        self.body_head        = nn.Linear(hidden_size, num_classes_dict["body"])
        self.contrast_head    = nn.Linear(hidden_size, num_classes_dict["contrast"])

        self.loss_fct = nn.CrossEntropyLoss()
        self.plane_class_weights: Optional[torch.Tensor] = None
        self.body_class_weights: Optional[torch.Tensor] = None
    
    
    def set_head_class_weights(
        self,
        plane_weights: Optional[torch.Tensor] = None,
        body_weights: Optional[torch.Tensor] = None,
           ):
            if plane_weights is not None:
                self.plane_class_weights = plane_weights
            if body_weights is not None:
                self.body_class_weights = body_weights


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels_modality: Optional[torch.Tensor] = None,
        labels_vendor: Optional[torch.Tensor] = None,
        labels_series_type: Optional[torch.Tensor] = None,
        labels_plane: Optional[torch.Tensor] = None,
        labels_acquisition: Optional[torch.Tensor] = None,
        labels_body: Optional[torch.Tensor] = None,
        labels_contrast: Optional[torch.Tensor] = None,
    ):
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = enc_out.pooler_output  # (batch, hidden_size)

        logits_modality    = self.modality_head(pooled)
        logits_vendor      = self.vendor_head(pooled)
        logits_series_type = self.series_type_head(pooled)
        logits_plane       = self.plane_head(pooled)
        logits_acquisition = self.acq_head(pooled)
        logits_body        = self.body_head(pooled)
        logits_contrast    = self.contrast_head(pooled)

        loss = None
        if labels_modality is not None:
            loss = 0.0
            loss = loss + self.loss_fct(logits_modality,    labels_modality)
            loss = loss + self.loss_fct(logits_vendor,      labels_vendor)
            loss = loss + self.loss_fct(logits_series_type, labels_series_type)

            plane_loss_fct = (
                nn.CrossEntropyLoss(weight=self.plane_class_weights)
                if self.plane_class_weights is not None
                else self.loss_fct
            )
            loss = loss + plane_loss_fct(logits_plane, labels_plane)

            loss = loss + self.loss_fct(logits_acquisition, labels_acquisition)

            body_loss_fct = (
                nn.CrossEntropyLoss(weight=self.body_class_weights)
                if self.body_class_weights is not None
                else self.loss_fct
            )
            loss = loss + body_loss_fct(logits_body, labels_body)

            loss = loss + self.loss_fct(logits_contrast, labels_contrast)

        return {
            "loss": loss,
            "logits_modality": logits_modality,
            "logits_vendor": logits_vendor,
            "logits_series_type": logits_series_type,
            "logits_plane": logits_plane,
            "logits_acquisition": logits_acquisition,
            "logits_body": logits_body,
            "logits_contrast": logits_contrast,
        }
