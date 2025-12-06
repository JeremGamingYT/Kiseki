#!/usr/bin/env python3
"""
Dataset Preprocessing Script for KISEKI

Converts video dataset (1TB) to streamable IVQ Index format.
Run once before training.

Usage:
    python scripts/preprocess_dataset.py \
        --input_dir /path/to/anime/videos \
        --output_dir ./data/ivq_index \
        --workers 8
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kiseki.tokenizer import AnimeVAE, AnimeVAEConfig
from kiseki.streaming.zero_copy import create_index_from_videos
from kiseki.streaming.ivq_index import IVQIndex, IndexConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess dataset for KISEKI')
    parser.add_argument('--input_dir', type=str, required=True, help='Input video directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output index directory')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for tokenization')
    parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for VAE encoding')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("KISEKI Dataset Preprocessor")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print()
    
    # Check input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        print("Creating demo dataset instead...")
        create_demo_dataset(args)
        return
    
    # Count videos
    video_extensions = ['.mp4', '.avi', '.mkv', '.webm', '.mov']
    videos = []
    for ext in video_extensions:
        videos.extend(input_path.rglob(f'*{ext}'))
    
    print(f"Found {len(videos)} video files")
    
    if len(videos) == 0:
        print("No videos found, creating demo dataset...")
        create_demo_dataset(args)
        return
    
    # Estimate total size
    total_size = sum(v.stat().st_size for v in videos)
    print(f"Total size: {total_size / 1024**3:.2f} GB")
    
    # Create VAE
    print("\nCreating VAE encoder...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    vae_config = AnimeVAEConfig(latent_dim=args.latent_dim, patch_size=args.patch_size)
    vae = AnimeVAE(vae_config).to(device).eval()
    
    # Process videos
    print("\nProcessing videos...")
    create_index_from_videos(
        video_dir=str(input_path),
        output_dir=args.output_dir,
        vae=vae,
        batch_size=args.batch_size,
        device=str(device),
    )
    
    # Create semantic index
    print("\nCreating semantic index...")
    create_semantic_index(args.output_dir, args.latent_dim)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Index saved to: {args.output_dir}")
    print("=" * 60)


def create_demo_dataset(args):
    """Create a demo dataset for testing"""
    import json
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating demo dataset...")
    
    # Generate random latents
    n_frames = 100000
    print(f"Generating {n_frames} random frames...")
    
    latents = np.random.randn(n_frames, args.latent_dim).astype(np.float16)
    
    # Save as binary
    latents_path = output_path / "latents.bin"
    latents.tofile(str(latents_path))
    print(f"Saved latents to {latents_path}")
    
    # Save metadata
    metadata = {
        'num_frames': n_frames,
        'num_samples': n_frames // 64,
        'latent_dim': args.latent_dim,
        'demo': True,
    }
    
    with open(output_path / "index.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created demo index with {n_frames} frames")
    
    # Create semantic index
    create_semantic_index(args.output_dir, args.latent_dim)


def create_semantic_index(output_dir: str, latent_dim: int):
    """Create IVQ index for semantic search"""
    import json
    
    output_path = Path(output_dir)
    
    # Load latents
    latents_path = output_path / "latents.bin"
    if not latents_path.exists():
        print("No latents found, skipping semantic index")
        return
    
    # Read a subset for training
    print("Training IVQ index...")
    
    # Read first 10000 vectors for training
    n_train = min(10000, 100000)
    latents = np.fromfile(str(latents_path), dtype=np.float16, count=n_train * latent_dim)
    latents = latents.reshape(-1, latent_dim).astype(np.float32)
    
    # Create and train index
    config = IndexConfig(dim=latent_dim, n_clusters=256)
    index = IVQIndex(config)
    index.train(latents)
    
    # Add all vectors (in batches for large datasets)
    print("Adding vectors to index...")
    batch_size = 10000
    
    total_vectors = latents.shape[0]
    for start in range(0, total_vectors, batch_size):
        end = min(start + batch_size, total_vectors)
        index.add(latents[start:end])
    
    # Save index
    ivq_path = output_path / "ivq_index"
    index.save(str(ivq_path))
    print(f"Saved IVQ index to {ivq_path}")


if __name__ == '__main__':
    main()
