from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseModel(ABC):
    """Abstract base class for all models in the fine-tuning framework."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.device = torch.device('cpu')

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def prepare_for_training(self) -> None:
        """Prepare the model for training (e.g., apply PEFT, move to device)."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass

    def to_device(self, device: torch.device) -> None:
        """Move model to specified device."""
        if self.model is not None:
            self.model.to(device)
        self.device = device

    def save_model(self, path: str) -> None:
        """Save the model."""
        if self.model is not None:
            self.model.save_pretrained(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)

    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get information about trainable parameters."""
        if self.model is None:
            return {}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
        }
