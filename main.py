#!/usr/bin/env python3
"""
Main script demonstrating the factory and strategy pattern implementation
for fine-tuning BioBERT with different PEFT strategies.
"""

import yaml
import argparse
from pathlib import Path

from src.factory.model_factory import ModelFactory
from src.factory.peft_factory import PEFTFactory
from src.trainer.trainer import FineTuneTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune BioBERT with PEFT strategies')
    parser.add_argument('--model_config', type=str, default='configs/model/biobert.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--peft_config', type=str, default='configs/peft/lora.yaml',
                       help='Path to PEFT configuration file')
    parser.add_argument('--training_config', type=str, default='configs/training.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--peft_type', type=str, choices=['lora', 'qlora'], default='lora',
                       help='Type of PEFT strategy to use')

    args = parser.parse_args()

    # Load configurations
    model_config = load_config(args.model_config)
    peft_config = load_config(args.peft_config) if Path(args.peft_config).exists() else {}
    training_config = load_config(args.training_config) if Path(args.training_config).exists() else {}

    print("=== Fine-tuning BioBERT with Factory and Strategy Patterns ===")
    print(f"Model: {model_config.get('name', 'Unknown')}")
    print(f"PEFT Strategy: {args.peft_type}")
    print()

    try:
        # Create model using factory pattern
        print("1. Creating model...")
        model = ModelFactory.create_model('biobert', model_config)
        model.load_model()
        print(f"   Model loaded: {model.get_model_info()['model_name']}")
        print()

        # Create PEFT strategy using factory pattern
        print("2. Creating PEFT strategy...")
        peft_strategy = PEFTFactory.create_peft(args.peft_type, peft_config)
        print(f"   PEFT strategy created: {peft_strategy.__class__.__name__}")
        print(f"   Config: {peft_strategy.get_peft_info()}")
        print()

        # Create trainer
        print("3. Setting up trainer...")
        trainer = FineTuneTrainer(model, peft_strategy, training_config)
        print("   Trainer created successfully")
        print()

        # Display training setup info
        print("4. Training setup information:")
        info = trainer.get_training_info()
        print(f"   Model parameters: {info['model_info'].get('parameters', {})}")
        if info['peft_info']:
            print(f"   PEFT parameters: {info['peft_info'].get('r', 'N/A')} rank")
        print()

        print("=== Setup Complete ===")
        print("To start training, you would typically:")
        print("1. Load your dataset")
        print("2. Call trainer.prepare_for_training(train_dataset, eval_dataset)")
        print("3. Call trainer.train()")
        print("4. Call trainer.save_model('path/to/save')")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
