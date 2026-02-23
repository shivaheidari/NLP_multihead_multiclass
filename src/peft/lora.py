from typing import Dict, Any
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel
from .base import BasePEFT


class LoRA(BasePEFT):
    """LoRA (Low-Rank Adaptation) PEFT strategy."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.r = config.get('r', 8)  # Rank
        self.lora_alpha = config.get('lora_alpha', 16)
        self.lora_dropout = config.get('lora_dropout', 0.1)
        self.target_modules = config.get('target_modules', ["query", "key", "value"])
        self.bias = config.get('bias', 'none')
        self.task_type = config.get('task_type', 'SEQ_CLS')  # For sequence classification

    def create_config(self) -> LoraConfig:
        """Create LoRA configuration."""
        self.peft_config = LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type
        )
        return self.peft_config

    def apply_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply LoRA to the model."""
        if self.peft_config is None:
            self.create_config()

        peft_model = get_peft_model(model, self.peft_config)
        self.is_applied = True
        print("LoRA applied to model successfully")
        return peft_model

    def get_peft_info(self) -> Dict[str, Any]:
        """Get LoRA configuration information."""
        return {
            'method': 'LoRA',
            'r': self.r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'target_modules': self.target_modules,
            'bias': self.bias,
            'task_type': self.task_type,
            'is_applied': self.is_applied
        }
