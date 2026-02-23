from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from transformers import PreTrainedModel


class BasePEFT(ABC):
    """Abstract base class for Parameter-Efficient Fine-Tuning strategies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.peft_config = None
        self.is_applied = False

    @abstractmethod
    def create_config(self) -> Any:
        """Create the PEFT configuration object."""
        pass

    @abstractmethod
    def apply_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply PEFT to the model."""
        pass

    @abstractmethod
    def get_peft_info(self) -> Dict[str, Any]:
        """Get information about the PEFT configuration."""
        pass

    def is_peft_applied(self) -> bool:
        """Check if PEFT has been applied."""
        return self.is_applied

    def save_peft_model(self, model: PreTrainedModel, path: str) -> None:
        """Save the PEFT model."""
        if not self.is_applied:
            raise RuntimeError("PEFT not applied to model. Call apply_peft() first.")

        model.save_pretrained(path)
        print(f"PEFT model saved to: {path}")

    def get_trainable_parameters(self, model: PreTrainedModel) -> Dict[str, int]:
        """Get trainable parameters after PEFT application."""
        if not self.is_applied:
            return {}

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
        }
