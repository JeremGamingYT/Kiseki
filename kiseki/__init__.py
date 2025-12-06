"""
Project KISEKI (奇跡) - Revolutionary Anime Video Generation

A neuro-symbolic latent architecture for anime video generation,
designed to run on consumer GPUs while processing 1TB+ datasets.

Core Innovations:
    - Zero-Copy Neural Streaming via NVMe
    - Mamba-Flow Architecture (SSM + Flow Matching)
    - SVG-Latent Representation (4000x compression)
    - Self-Improvement Training Loop
"""

__version__ = "0.1.0"
__author__ = "KISEKI Team"
__license__ = "MIT"

from kiseki.core import MambaFlowGenerator, StateMemory, SparseLocalAttention
from kiseki.tokenizer import AnimeVAE, SVGLatentSpace
from kiseki.streaming import ZeroCopyLoader, IVQIndex

__all__ = [
    "MambaFlowGenerator",
    "StateMemory", 
    "SparseLocalAttention",
    "AnimeVAE",
    "SVGLatentSpace",
    "ZeroCopyLoader",
    "IVQIndex",
]
