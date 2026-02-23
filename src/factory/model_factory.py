from typing import Dict, Any
from ..models.base import BaseModel
from ..models.hf_model import HuggingFaceModel


class ModelFactory:
    """Factory for creating model instances."""

    _models = {
        'huggingface': HuggingFaceModel,
        'biobert': HuggingFaceModel,  # BioBERT uses HuggingFace implementation
    }

    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance.

        Args:
            model_type: Type of model ('huggingface', 'biobert', etc.)
            config: Configuration dictionary for the model

        Returns:
            BaseModel: Model instance

        Raises:
            ValueError: If model_type is not supported
        """
        model_type = model_type.lower()
        if model_type not in ModelFactory._models:
            available = list(ModelFactory._models.keys())
            raise ValueError(f"Unsupported model type: {model_type}. Available: {available}")

        model_class = ModelFactory._models[model_type]
        return model_class(config)

    @staticmethod
    def get_available_model_types() -> list:
        """Get list of available model types."""
        return list(ModelFactory._models.keys())

    @staticmethod
    def register_model_type(model_type: str, model_class: type) -> None:
        """
        Register a new model type.

        Args:
            model_type: Name of the model type
            model_class: Class implementing the model
        """
        ModelFactory._models[model_type.lower()] = model_class
