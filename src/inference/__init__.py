"""
Inference Module — Multilingual Speech Recognition Inference.
"""

from .inference import STTInference
from .post_processor import STTPostProcessor

__all__ = ["STTInference", "STTPostProcessor"]
