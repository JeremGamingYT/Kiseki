"""
Self-Improvement Loop - Exponential Training for KISEKI

The magic that makes training exponentially faster:
1. Generate synthetic clips
2. Judge quality with CLIP-Anime discriminator  
3. Add good clips to training data
4. Model learns from its own successes
"""

import random
from typing import Optional, Dict, List, Tuple
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SelfImproveConfig:
    """Configuration for self-improvement loop"""
    buffer_size: int = 10000
    quality_threshold: float = 0.7
    synthetic_ratio: float = 0.2  # Ratio of synthetic data in batch
    update_interval: int = 100  # Steps between synthetic generation
    max_synthetic_per_step: int = 16


class SyntheticDataBuffer:
    """
    Buffer for storing high-quality synthetic samples.
    Uses priority queue based on quality scores.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = []  # (quality_score, sample) tuples
    
    def add(self, sample: Dict[str, torch.Tensor], quality: float):
        """Add sample if quality is high enough"""
        if len(self.buffer) >= self.max_size:
            # Remove lowest quality sample
            min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i][0])
            if quality > self.buffer[min_idx][0]:
                self.buffer[min_idx] = (quality, sample)
        else:
            self.buffer.append((quality, sample))
    
    def sample(self, n: int) -> List[Dict[str, torch.Tensor]]:
        """Sample n items from buffer"""
        if not self.buffer:
            return []
        n = min(n, len(self.buffer))
        # Weighted sampling by quality
        weights = [q for q, _ in self.buffer]
        total = sum(weights)
        probs = [w / total for w in weights]
        indices = random.choices(range(len(self.buffer)), weights=probs, k=n)
        return [self.buffer[i][1] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    @property
    def mean_quality(self) -> float:
        if not self.buffer:
            return 0.0
        return sum(q for q, _ in self.buffer) / len(self.buffer)


class CLIPAnimeDiscriminator(nn.Module):
    """
    CLIP-based quality discriminator specialized for anime.
    
    Judges generated clips on:
    - Visual quality
    - Style consistency
    - Motion smoothness
    - Character coherence
    """
    
    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Frame encoder (simplified - real version uses CLIP)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Temporal aggregation
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Quality heads
        self.visual_quality = nn.Linear(hidden_dim, 1)
        self.style_consistency = nn.Linear(hidden_dim, 1)
        self.motion_smoothness = nn.Linear(hidden_dim, 1)
        self.character_coherence = nn.Linear(hidden_dim, 1)
        
        # Final quality score
        self.quality_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: Latent frames [batch, seq_len, dim]
        
        Returns:
            Quality scores dictionary
        """
        batch, seq_len, _ = frames.shape
        
        # Encode frames
        h = self.encoder(frames)  # [batch, seq, hidden]
        
        # Temporal aggregation
        _, final_state = self.temporal(h)
        final_state = final_state.squeeze(0)  # [batch, hidden]
        
        # Compute subscores
        visual = torch.sigmoid(self.visual_quality(final_state))
        style = torch.sigmoid(self.style_consistency(final_state))
        
        # Motion smoothness from frame differences
        diffs = (frames[:, 1:] - frames[:, :-1]).abs().mean(dim=(1, 2))
        motion = torch.sigmoid(-diffs + 0.5).unsqueeze(-1)  # Smooth = low diff
        
        # Character coherence from frame similarity
        first_frame = frames[:, 0]
        last_frame = frames[:, -1]
        coherence = F.cosine_similarity(first_frame, last_frame).unsqueeze(-1)
        coherence = (coherence + 1) / 2  # Map to [0, 1]
        
        # Aggregate scores
        subscores = torch.cat([visual, style, motion, coherence], dim=-1)
        final_quality = self.quality_head(subscores)
        
        return {
            'quality': final_quality.squeeze(-1),
            'visual': visual.squeeze(-1),
            'style': style.squeeze(-1),
            'motion': motion.squeeze(-1),
            'coherence': coherence.squeeze(-1),
        }
    
    def compute_loss(
        self,
        real_frames: torch.Tensor,
        fake_frames: torch.Tensor,
    ) -> torch.Tensor:
        """Discriminator loss (real vs fake)"""
        real_scores = self(real_frames)['quality']
        fake_scores = self(fake_frames)['quality']
        
        real_loss = F.binary_cross_entropy(real_scores, torch.ones_like(real_scores))
        fake_loss = F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores))
        
        return (real_loss + fake_loss) / 2


class SelfImproveLoop:
    """
    Self-Improvement Training Loop for KISEKI.
    
    The key insight: Use the model's best generations
    as additional training data, creating a positive
    feedback loop that accelerates learning.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: Optional[nn.Module] = None,
        config: Optional[SelfImproveConfig] = None,
    ):
        self.generator = generator
        self.discriminator = discriminator or CLIPAnimeDiscriminator()
        self.config = config or SelfImproveConfig()
        
        self.buffer = SyntheticDataBuffer(self.config.buffer_size)
        self.step_counter = 0
        
        # Statistics
        self.stats = {
            'samples_generated': 0,
            'samples_accepted': 0,
            'mean_quality': 0.0,
        }
    
    @torch.no_grad()
    def generate_synthetic(
        self,
        n_samples: int,
        seq_len: int = 64,
        latent_dim: int = 512,
        device: torch.device = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate synthetic samples and filter by quality"""
        device = device or next(self.generator.parameters()).device
        
        accepted = []
        
        for _ in range(n_samples):
            # Generate
            generated = self.generator.generate(
                shape=(1, seq_len, latent_dim),
                n_steps=20,
                device=device,
            )
            
            # Judge quality
            quality_scores = self.discriminator(generated)
            quality = quality_scores['quality'].item()
            
            self.stats['samples_generated'] += 1
            
            if quality >= self.config.quality_threshold:
                sample = {
                    'latents': generated.squeeze(0).cpu(),
                    'quality': quality,
                }
                accepted.append(sample)
                self.buffer.add(sample, quality)
                self.stats['samples_accepted'] += 1
        
        self.stats['mean_quality'] = self.buffer.mean_quality
        return accepted
    
    def get_mixed_batch(
        self,
        real_batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Mix real and synthetic data for training"""
        batch_size = real_batch['latents'].shape[0]
        n_synthetic = int(batch_size * self.config.synthetic_ratio)
        
        if len(self.buffer) == 0 or n_synthetic == 0:
            return real_batch
        
        # Sample from buffer
        synthetic = self.buffer.sample(n_synthetic)
        if not synthetic:
            return real_batch
        
        synthetic_latents = torch.stack([s['latents'] for s in synthetic])
        synthetic_latents = synthetic_latents.to(device)
        
        # Mix with real data
        n_real = batch_size - n_synthetic
        mixed_latents = torch.cat([
            real_batch['latents'][:n_real],
            synthetic_latents,
        ], dim=0)
        
        return {'latents': mixed_latents}
    
    def step(
        self,
        real_batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """One training step with self-improvement"""
        self.step_counter += 1
        
        # Periodically generate synthetic samples
        if self.step_counter % self.config.update_interval == 0:
            seq_len = real_batch['latents'].shape[1]
            latent_dim = real_batch['latents'].shape[2]
            self.generate_synthetic(
                self.config.max_synthetic_per_step,
                seq_len=seq_len,
                latent_dim=latent_dim,
                device=device,
            )
        
        # Get mixed batch
        mixed_batch = self.get_mixed_batch(real_batch, device)
        
        return mixed_batch
    
    def get_stats(self) -> Dict:
        """Get training statistics"""
        return {
            **self.stats,
            'buffer_size': len(self.buffer),
            'acceptance_rate': (
                self.stats['samples_accepted'] / max(1, self.stats['samples_generated'])
            ),
        }


class TeacherStudentDistillation:
    """
    Inverted Teacher-Student distillation for KISEKI.
    
    Unlike traditional distillation (big model → small model),
    we distill knowledge from successful generations back
    into the model itself:
    
    Current Model → Generates → Best Clips → Teaches → Current Model
    """
    
    def __init__(
        self,
        model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        self.model = model
        self.temperature = temperature
        self.alpha = alpha
        
        # Store teacher targets (from best generations)
        self.teacher_targets = []
    
    def add_teacher_target(
        self,
        input_latents: torch.Tensor,
        output_velocity: torch.Tensor,
        timestep: torch.Tensor,
    ):
        """Store a successful generation as teacher target"""
        target = {
            'input': input_latents.detach().cpu(),
            'output': output_velocity.detach().cpu(),
            't': timestep.detach().cpu(),
        }
        self.teacher_targets.append(target)
        
        # Keep buffer bounded
        if len(self.teacher_targets) > 1000:
            self.teacher_targets.pop(0)
    
    def distillation_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss"""
        # Soft targets with temperature
        student_soft = student_output / self.temperature
        teacher_soft = teacher_output / self.temperature
        
        loss = F.mse_loss(student_soft, teacher_soft)
        return loss * (self.temperature ** 2)
    
    def get_distillation_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Get batch of teacher targets for distillation"""
        if len(self.teacher_targets) < batch_size:
            return None
        
        indices = random.sample(range(len(self.teacher_targets)), batch_size)
        batch = {
            'input': torch.stack([self.teacher_targets[i]['input'] for i in indices]),
            'output': torch.stack([self.teacher_targets[i]['output'] for i in indices]),
            't': torch.stack([self.teacher_targets[i]['t'] for i in indices]),
        }
        return {k: v.to(device) for k, v in batch.items()}


if __name__ == "__main__":
    # Test self-improvement loop
    from kiseki.core import MambaFlowGenerator, MambaFlowConfig
    
    config = MambaFlowConfig(d_model=128, n_layers=2, latent_dim=64)
    generator = MambaFlowGenerator(config)
    
    loop = SelfImproveLoop(generator)
    
    # Simulate training step
    fake_batch = {'latents': torch.randn(8, 32, 64)}
    mixed = loop.step(fake_batch, torch.device('cpu'))
    
    print(f"Original batch shape: {fake_batch['latents'].shape}")
    print(f"Mixed batch shape: {mixed['latents'].shape}")
    print(f"Stats: {loop.get_stats()}")
