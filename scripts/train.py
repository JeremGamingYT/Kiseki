#!/usr/bin/env python3
"""
KISEKI Training Script

Main training loop for the KISEKI anime video generator.
Supports single GPU training on RTX 3090/4090.

Usage:
    python scripts/train.py --config configs/model_600m.yaml

Features:
    - Zero-Copy data streaming
    - Flow Matching objective
    - Self-improvement loop
    - Physics priors
    - Mixed precision training
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Dict

import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kiseki.core import MambaFlowGenerator
from kiseki.core.mamba_flow import MambaFlowConfig
from kiseki.tokenizer import AnimeVAE, AnimeVAEConfig
from kiseki.streaming import ZeroCopyLoader, StreamConfig
from kiseki.training import (
    FlowMatchingLoss,
    SelfImproveLoop, 
    SelfImproveConfig,
    AnimePriors,
    PhysicsConfig,
)
from kiseki.utils import MemoryTracker, gradient_checkpointing


def parse_args():
    parser = argparse.ArgumentParser(description='Train KISEKI model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed')
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(config: Dict) -> nn.Module:
    """Create KISEKI model from config"""
    model_config = MambaFlowConfig(
        d_model=config['model']['d_model'],
        d_state=config['model']['d_state'],
        d_conv=config['model']['d_conv'],
        expand_factor=config['model']['expand_factor'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        latent_dim=config['model']['latent_dim'],
        flow_time_embed_dim=config['model']['flow_time_embed_dim'],
        dropout=config['model']['dropout'],
        use_flash_attention=config['model']['use_flash_attention'],
        gradient_checkpointing=config['model']['gradient_checkpointing'],
    )
    
    model = MambaFlowGenerator(model_config)
    
    if config['model']['gradient_checkpointing']:
        gradient_checkpointing(model, enable=True)
    
    return model


def create_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Create optimizer from config"""
    train_config = config['training']
    
    # Separate weight decay for different param types
    no_decay = ['bias', 'LayerNorm', 'layer_norm', 'norm']
    
    params = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': train_config['weight_decay'],
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        params,
        lr=train_config['learning_rate'],
        betas=tuple(train_config['betas']),
    )
    
    return optimizer


def create_scheduler(optimizer, config: Dict, total_steps: int):
    """Create learning rate scheduler"""
    train_config = config['training']
    warmup_steps = train_config['warmup_steps']
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_fn: FlowMatchingLoss,
    physics_priors: Optional[AnimePriors],
    scaler: GradScaler,
    config: Dict,
    device: torch.device,
) -> Dict[str, float]:
    """Single training step"""
    
    # Get data
    latents = batch['latents'].to(device)
    
    # Forward pass with mixed precision
    with autocast(enabled=config['training']['fp16']):
        # Flow matching loss
        fm_loss = loss_fn(model, latents, return_components=False)
        
        # Physics priors loss
        if physics_priors is not None:
            physics_loss = physics_priors.get_loss(latents)
            total_loss = fm_loss + 0.1 * physics_loss
        else:
            physics_loss = torch.tensor(0.0, device=device)
            total_loss = fm_loss
    
    return {
        'loss': total_loss,
        'fm_loss': fm_loss.item(),
        'physics_loss': physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: Dict,
    output_dir: Path,
):
    """Save training checkpoint"""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
    }
    
    path = output_dir / f'checkpoint-{step}.pt'
    torch.save(checkpoint, path)
    print(f'Saved checkpoint to {path}')
    
    # Clean up old checkpoints
    checkpoints = sorted(output_dir.glob('checkpoint-*.pt'))
    save_top_k = config['checkpoint'].get('save_top_k', 3)
    for old_ckpt in checkpoints[:-save_top_k]:
        old_ckpt.unlink()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training KISEKI on {device}")
    print(f"Config: {args.config}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Memory tracker
    tracker = MemoryTracker()
    print(tracker.summary())
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    num_params = model.get_num_params()
    print(f"Model parameters: {num_params:,}")
    tracker.log("After model creation")
    print(tracker.summary())
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    total_steps = config['training']['max_steps']
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    # Create loss function
    loss_fn = FlowMatchingLoss()
    
    # Create physics priors
    if config['training']['physics']['enabled']:
        physics_config = PhysicsConfig(
            hair_weight=config['training']['physics']['hair_weight'],
            eye_weight=config['training']['physics']['eye_weight'],
            motion_weight=config['training']['physics']['motion_weight'],
        )
        physics_priors = AnimePriors(config['model']['latent_dim'], physics_config)
        physics_priors = physics_priors.to(device)
    else:
        physics_priors = None
    
    # Create self-improvement loop
    if config['training']['self_improve']['enabled']:
        si_config = SelfImproveConfig(
            buffer_size=config['training']['self_improve']['buffer_size'],
            quality_threshold=config['training']['self_improve']['quality_threshold'],
            synthetic_ratio=config['training']['self_improve']['synthetic_ratio'],
            update_interval=config['training']['self_improve']['update_interval'],
        )
        self_improve = SelfImproveLoop(model, config=si_config)
    else:
        self_improve = None
    
    # Create data loader
    print("Creating data loader...")
    data_config = config['data']
    
    # Check if index exists, otherwise create demo data
    index_path = Path(data_config['index_path'])
    if not index_path.exists():
        print(f"Index not found at {index_path}, creating demo data...")
        index_path.mkdir(parents=True, exist_ok=True)
    
    stream_config = StreamConfig(
        index_path=str(index_path),
        batch_size=config['training']['batch_size'],
        num_workers=data_config['num_workers'],
        prefetch_factor=data_config['prefetch_factor'],
        pin_memory=data_config['pin_memory'],
    )
    
    loader = ZeroCopyLoader(
        stream_config,
        latent_dim=config['model']['latent_dim'],
        sequence_length=data_config['sequence_length'],
    )
    
    print("Memory usage estimate:")
    for k, v in loader.estimate_memory_usage().items():
        print(f"  {k}: {v:.2f} MB")
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config['training']['fp16'])
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step']
    
    # Training loop
    print(f"\nStarting training from step {start_step}...")
    print(f"Total steps: {total_steps}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation']}")
    print(f"Effective batch size: {config['training']['effective_batch_size']}")
    
    model.train()
    grad_accum = config['training']['gradient_accumulation']
    log_interval = config['logging']['log_interval']
    save_interval = config['logging']['save_interval']
    
    step = start_step
    epoch = 0
    
    running_loss = 0.0
    running_fm_loss = 0.0
    running_physics_loss = 0.0
    
    start_time = time.time()
    
    while step < total_steps:
        epoch += 1
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            # Self-improvement: potentially mix synthetic data
            if self_improve is not None:
                batch = self_improve.step(batch, device)
            
            # Training step
            losses = train_step(
                model, batch, loss_fn, physics_priors,
                scaler, config, device
            )
            
            loss = losses['loss'] / grad_accum
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Accumulation step
            if (step + 1) % grad_accum == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['max_grad_norm']
                )
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            # Update running stats
            running_loss += losses['loss'].item()
            running_fm_loss += losses['fm_loss']
            running_physics_loss += losses['physics_loss']
            
            step += 1
            
            # Logging
            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                avg_fm = running_fm_loss / log_interval
                avg_physics = running_physics_loss / log_interval
                lr = scheduler.get_last_lr()[0]
                
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'fm': f'{avg_fm:.4f}',
                    'lr': f'{lr:.2e}',
                    'step/s': f'{steps_per_sec:.2f}',
                })
                
                running_loss = 0.0
                running_fm_loss = 0.0
                running_physics_loss = 0.0
                
                if self_improve is not None:
                    stats = self_improve.get_stats()
                    if stats['samples_generated'] > 0:
                        print(f"  Self-improve: {stats['acceptance_rate']:.1%} accepted, "
                              f"buffer: {stats['buffer_size']}")
            
            # Save checkpoint
            if step % save_interval == 0:
                save_checkpoint(model, optimizer, scheduler, step, config, output_dir)
            
            if step >= total_steps:
                break
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, step, config, output_dir)
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Total time: {elapsed / 3600:.2f} hours")
    print(f"Final step: {step}")
    print(tracker.summary())


if __name__ == '__main__':
    main()
