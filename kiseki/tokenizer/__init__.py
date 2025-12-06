"""
KISEKI Tokenizer - Neural Tokenizer for Anime

Specialized VAE and SVG-Latent representation:
- AnimeVAE: Separates lineart from color
- SVGLatentSpace: 4000x compression via vector primitives
"""

from kiseki.tokenizer.anime_vae import AnimeVAE, AnimeVAEConfig
from kiseki.tokenizer.svg_latent import SVGLatentSpace, SVGPrimitive

__all__ = ["AnimeVAE", "AnimeVAEConfig", "SVGLatentSpace", "SVGPrimitive"]
