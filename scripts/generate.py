#!/usr/bin/env python3
"""
KISEKI Generation Script

Generate anime clips using a trained KISEKI model.

Usage:
    python scripts/generate.py \
        --checkpoint ./checkpoints/kiseki_600m.pt \
        --prompt "A girl with blue hair running through cherry blossoms" \
        --duration 5.0 \
        --output ./output.mp4
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kiseki.core import MambaFlowGenerator
from kiseki.core.mamba_flow import MambaFlowConfig
from kiseki.tokenizer import AnimeVAE, SVGLatentSpace
from kiseki.training import FlowMatchingSampler


def parse_args():
    parser = argparse.ArgumentParser(description='Generate anime with KISEKI')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt (optional)')
    parser.add_argument('--duration', type=float, default=5.0, help='Duration in seconds')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second')
    parser.add_argument('--output', type=str, default='./output.mp4', help='Output path')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--cfg_scale', type=float, default=7.5, help='CFG scale')
    parser.add_argument('--n_steps', type=int, default=50, help='Sampling steps')
    parser.add_argument('--resolution', type=int, default=512, help='Output resolution')
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create model config
    model_cfg = config.get('model', {})
    model_config = MambaFlowConfig(
        d_model=model_cfg.get('d_model', 1024),
        d_state=model_cfg.get('d_state', 64),
        n_layers=model_cfg.get('n_layers', 24),
        n_heads=model_cfg.get('n_heads', 16),
        latent_dim=model_cfg.get('latent_dim', 512),
    )
    
    model = MambaFlowGenerator(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, model_config


def encode_prompt(prompt: str, device: torch.device) -> Optional[torch.Tensor]:
    """Encode text prompt to conditioning tensor"""
    if prompt is None:
        return None
    
    # In production, use CLIP or T5 encoder
    # For now, return a random embedding as placeholder
    print(f"Prompt: {prompt}")
    print("(Using placeholder embeddings - integrate CLIP for real prompts)")
    
    # Placeholder: 77 tokens, 768-dim
    return torch.randn(1, 77, 768, device=device)


def decode_latents_to_video(
    latents: torch.Tensor,
    vae: AnimeVAE,
    svg_space: SVGLatentSpace,
    device: torch.device,
) -> np.ndarray:
    """Decode latent sequence to video frames"""
    batch, seq_len, latent_dim = latents.shape
    
    frames = []
    for i in range(seq_len):
        z = latents[:, i]  # [batch, latent_dim]
        
        # Decode through SVG space
        svg_output = svg_space(z)
        rendered = svg_output['rendered']  # [batch, 3, H, W]
        
        # Convert to numpy
        frame = rendered[0].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frames.append(frame)
    
    return np.stack(frames)


def save_video(frames: np.ndarray, output_path: str, fps: int = 24):
    """Save frames as video"""
    try:
        import imageio
        print(f"Saving video to {output_path}")
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264')
        print(f"Video saved: {len(frames)} frames at {fps} FPS")
    except ImportError:
        # Fallback: save as GIF
        output_path = output_path.replace('.mp4', '.gif')
        import imageio
        imageio.mimwrite(output_path, frames, fps=fps)
        print(f"Saved as GIF (install imageio-ffmpeg for MP4): {output_path}")


def main():
    args = parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Calculate frame count
    n_frames = int(args.duration * args.fps)
    print(f"Generating {n_frames} frames ({args.duration}s at {args.fps} FPS)")
    
    # Load model
    if Path(args.checkpoint).exists():
        model, model_config = load_model(args.checkpoint, device)
        latent_dim = model_config.latent_dim
    else:
        # Demo mode: create untrained model
        print("Checkpoint not found, running in demo mode...")
        model_config = MambaFlowConfig(d_model=256, n_layers=4, latent_dim=256)
        model = MambaFlowGenerator(model_config)
        model = model.to(device).eval()
        latent_dim = 256
    
    # Create sampler
    sampler = FlowMatchingSampler(
        model,
        n_steps=args.n_steps,
        cfg_scale=args.cfg_scale,
    )
    
    # Encode prompt
    condition = encode_prompt(args.prompt, device)
    
    # Generate latents
    print(f"\nGenerating with {args.n_steps} steps, CFG scale {args.cfg_scale}...")
    with torch.no_grad():
        latents = sampler.sample_heun(
            shape=(1, n_frames, latent_dim),
            condition=condition,
            device=device,
        )
    
    print(f"Generated latents shape: {latents.shape}")
    
    # Decode to video
    print("\nDecoding to video...")
    
    # Create decoder modules
    from kiseki.tokenizer import AnimeVAEConfig
    vae_config = AnimeVAEConfig(latent_dim=latent_dim, img_size=args.resolution)
    vae = AnimeVAE(vae_config).to(device).eval()
    svg_space = SVGLatentSpace(latent_dim=latent_dim, img_size=args.resolution).to(device).eval()
    
    with torch.no_grad():
        frames = decode_latents_to_video(latents, vae, svg_space, device)
    
    print(f"Decoded {len(frames)} frames, shape: {frames[0].shape}")
    
    # Save video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(frames, str(output_path), fps=args.fps)
    
    print("\nGeneration complete!")


if __name__ == '__main__':
    main()
