from typing import Dict, Any
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import PreTrainedModel, BitsAndBytesConfig
from .base import BasePEFT


class QLoRA(BasePEFT):
    """QLoRA (Quantized Low-Rank Adaptation) PEFT strategy."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.r = config.get('r', 8)  # Rank
        self.lora_alpha = config.get('lora_alpha', 16)
        self.lora_dropout = config.get('lora_dropout', 0.1)
        self.target_modules = config.get('target_modules', ["query", "key", "value"])
        self.bias = config.get('bias', 'none')
        self.task_type = config.get('task_type', 'SEQ_CLS')  # For sequence classification

        # Quantization config
        self.load_in_4bit = config.get('load_in_4bit', True)
        self.load_in_8bit = config.get('load_in_8bit', False)
        self.llm_int8_threshold = config.get('llm_int8_threshold', 6.0)
        self.llm_int8_has_fp16_weight = config.get('llm_int8_has_fp16_weight', False)
        self.bnb_4bit_compute_dtype = getattr(torch, config.get('bnb_4bit_compute_dtype', 'float16'))
        self.bnb_4bit_use_double_quant = config.get('bnb_4bit_use_double_quant', True)
        self.bnb_4bit_quant_type = config.get('bnb_4bit_quant_type', 'nf4')

    def create_config(self) -> LoraConfig:
        """Create QLoRA configuration with quantization."""
        # Create quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            llm_int8_threshold=self.llm_int8_threshold,
            llm_int8_has_fp16_weight=self.llm_int8_has_fp16_weight,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type
        )

        # Create LoRA config
        self.peft_config = LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type,
            quantization_config=quantization_config
        )
        return self.peft_config

    def apply_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply QLoRA to the model with quantization."""
        if self.peft_config is None:
            self.create_config()

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Apply PEFT
        peft_model = get_peft_model(model, self.peft_config)
        self.is_applied = True
        print("QLoRA applied to model successfully")
        return peft_model

    def get_peft_info(self) -> Dict[str, Any]:
        """Get QLoRA configuration information."""
        return {
            'method': 'QLoRA',
            'r': self.r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'target_modules': self.target_modules,
            'bias': self.bias,
            'task_type': self.task_type,
            'quantization': {
                'load_in_4bit': self.load_in_4bit,
                'load_in_8bit': self.load_in_8bit,
                'bnb_4bit_compute_dtype': str(self.bnb_4bit_compute_dtype),
                'bnb_4bit_use_double_quant': self.bnb_4bit_use_double_quant,
                'bnb_4bit_quant_type': self.bnb_4bit_quant_type
            },
            'is_applied': self.is_applied
        }
