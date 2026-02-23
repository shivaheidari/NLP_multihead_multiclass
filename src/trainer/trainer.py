from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from ..models.base import BaseModel
from ..peft.base import BasePEFT


class FineTuneTrainer:
    """Trainer class for fine-tuning models with PEFT strategies."""

    def __init__(self,
                 model: BaseModel,
                 peft_strategy: Optional[BasePEFT] = None,
                 training_config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.peft_strategy = peft_strategy
        self.training_config = training_config or {}
        self.trainer: Optional[Trainer] = None
        self.is_prepared = False

    def prepare_for_training(self, train_dataset=None, eval_dataset=None) -> None:
        """Prepare the model and trainer for training."""
        if self.model.model is None:
            raise RuntimeError("Model not loaded. Call model.load_model() first.")

        # Apply PEFT if specified
        if self.peft_strategy is not None:
            print(f"Applying {self.peft_strategy.__class__.__name__}...")
            self.model.model = self.peft_strategy.apply_peft(self.model.model)

        # Prepare model for training
        self.model.prepare_for_training()

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.get('output_dir', './outputs'),
            num_train_epochs=self.training_config.get('num_train_epochs', 3),
            per_device_train_batch_size=self.training_config.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=self.training_config.get('per_device_eval_batch_size', 8),
            warmup_steps=self.training_config.get('warmup_steps', 500),
            weight_decay=self.training_config.get('weight_decay', 0.01),
            logging_dir=self.training_config.get('logging_dir', './logs'),
            logging_steps=self.training_config.get('logging_steps', 10),
            evaluation_strategy=self.training_config.get('evaluation_strategy', 'steps'),
            eval_steps=self.training_config.get('eval_steps', 500),
            save_steps=self.training_config.get('save_steps', 500),
            save_total_limit=self.training_config.get('save_total_limit', 3),
            load_best_model_at_end=self.training_config.get('load_best_model_at_end', True),
            metric_for_best_model=self.training_config.get('metric_for_best_model', 'eval_loss'),
            greater_is_better=self.training_config.get('greater_is_better', False),
            fp16=self.training_config.get('fp16', True),
            gradient_checkpointing=self.training_config.get('gradient_checkpointing', False),
        )

        # Create data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.model.tokenizer)

        # Create trainer
        self.trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.model.tokenizer,
            data_collator=data_collator,
        )

        self.is_prepared = True
        print("Trainer prepared for training")

    def train(self) -> None:
        """Start training."""
        if not self.is_prepared:
            raise RuntimeError("Trainer not prepared. Call prepare_for_training() first.")

        print("Starting training...")
        self.trainer.train()
        print("Training completed")

    def evaluate(self, eval_dataset=None) -> Dict[str, float]:
        """Evaluate the model."""
        if not self.is_prepared:
            raise RuntimeError("Trainer not prepared. Call prepare_for_training() first.")

        print("Evaluating model...")
        if eval_dataset is not None:
            results = self.trainer.evaluate(eval_dataset=eval_dataset)
        else:
            results = self.trainer.evaluate()

        print(f"Evaluation results: {results}")
        return results

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        if not self.is_prepared:
            raise RuntimeError("Trainer not prepared. Call prepare_for_training() first.")

        if self.peft_strategy is not None and self.peft_strategy.is_peft_applied():
            self.peft_strategy.save_peft_model(self.model.model, path)
        else:
            self.model.save_model(path)

        print(f"Model saved to: {path}")

    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training setup."""
        peft_info = {}
        if self.peft_strategy is not None:
            peft_info = self.peft_strategy.get_peft_info()

        return {
            'model_info': self.model.get_model_info(),
            'peft_info': peft_info,
            'training_config': self.training_config,
            'is_prepared': self.is_prepared
        }
