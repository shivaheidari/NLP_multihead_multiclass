from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from .base import BaseModel


class HuggingFaceModel(BaseModel):
    """HuggingFace model implementation for sequence classification tasks."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('name', '')
        self.tokenizer_path = config.get('tokenizer', '')
        self.dtype = getattr(torch, config.get('dtype', 'float32'))
        self.trust_remote_code = config.get('trust_remote_code', False)
        self.num_labels = config.get('num_labels', 2)  # Default for binary classification

    def load_model(self) -> None:
        """Load the HuggingFace model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=self.trust_remote_code
            )

            # Load model configuration
            model_config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                num_labels=self.num_labels
            )

            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=model_config,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.dtype
            )

            print(f"Successfully loaded model: {self.model_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")

    def prepare_for_training(self) -> None:
        """Prepare the model for training."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Move to device if specified
        if hasattr(self, 'device'):
            self.model.to(self.device)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        if self.model is None:
            return {}

        return {
            'model_name': self.model_name,
            'tokenizer_path': self.tokenizer_path,
            'dtype': str(self.dtype),
            'num_labels': self.num_labels,
            'model_type': type(self.model).__name__,
            'config': self.model.config.to_dict() if hasattr(self.model, 'config') else {},
            'parameters': self.get_trainable_parameters()
        }
