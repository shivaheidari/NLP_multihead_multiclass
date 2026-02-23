from typing import Dict, Any
from .base import BasePEFT
from .lora import LoRA
from .qlora import QLoRA


class PEFTFactory:
    """Factory for creating PEFT strategy instances."""

    _strategies = {
        'lora': LoRA,
        'qlora': QLoRA
    }

    @staticmethod
    def create_peft(peft_type: str, config: Dict[str, Any]) -> BasePEFT:
        """
        Create a PEFT strategy instance.

        Args:
            peft_type: Type of PEFT ('lora' or 'qlora')
            config: Configuration dictionary for the PEFT strategy

        Returns:
            BasePEFT: PEFT strategy instance

        Raises:
            ValueError: If peft_type is not supported
        """
        peft_type = peft_type.lower()
        if peft_type not in PEFTFactory._strategies:
            available = list(PEFTFactory._strategies.keys())
            raise ValueError(f"Unsupported PEFT type: {peft_type}. Available: {available}")

        strategy_class = PEFTFactory._strategies[peft_type]
        return strategy_class(config)

    @staticmethod
    def get_available_peft_types() -> list:
        """Get list of available PEFT types."""
        return list(PEFTFactory._strategies.keys())

    @staticmethod
    def register_peft_type(peft_type: str, strategy_class: type) -> None:
        """
        Register a new PEFT strategy.

        Args:
            peft_type: Name of the PEFT type
            strategy_class: Class implementing the PEFT strategy
        """
        PEFTFactory._strategies[peft_type.lower()] = strategy_class
