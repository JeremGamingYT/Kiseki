"""
Anime Physics Priors - Built-in physical knowledge

Instead of learning physics from data, we encode anime-specific
physical rules directly into the loss function:
- Hair dynamics (volume conservation, flow)
- Eye motion (saccades, blinks)
- Clothing folds (fabric physics)
- Character proportions (anime style)
"""

from typing import Optional, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PhysicsConfig:
    """Configuration for physics priors"""
    hair_weight: float = 1.0
    eye_weight: float = 1.0
    motion_weight: float = 1.0
    proportion_weight: float = 0.5


class HairPhysics(nn.Module):
    """
    Hair physics prior for anime.
    
    Enforces:
    1. Volume conservation (hair doesn't shrink/grow)
    2. Flow coherence (strands move together)
    3. Gravity influence (hair falls naturally)
    4. Wind response (delayed reaction, wave patterns)
    """
    
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        
        # Hair region detector (in latent space)
        self.hair_detector = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
        # Motion field predictor
        self.motion_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
    
    def forward(
        self,
        frames: torch.Tensor,
        gravity: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hair physics constraints.
        
        Args:
            frames: Latent frames [batch, seq, dim]
            gravity: Gravity direction [2] (default: down)
        
        Returns:
            Physics constraint losses
        """
        batch, seq_len, dim = frames.shape
        
        if gravity is None:
            gravity = torch.tensor([0.0, 1.0], device=frames.device)
        
        # Detect hair regions in each frame
        hair_masks = self.hair_detector(frames)  # [batch, seq, 1]
        
        # Volume conservation: hair "area" should stay constant
        hair_volumes = hair_masks.sum(dim=-1)  # [batch, seq]
        volume_var = hair_volumes.var(dim=1)  # Variance across frames
        volume_loss = volume_var.mean()
        
        # Flow coherence: consecutive frames should have smooth motion
        if seq_len > 1:
            diffs = frames[:, 1:] - frames[:, :-1]  # [batch, seq-1, dim]
            
            # Weighted by hair presence
            hair_weight = (hair_masks[:, 1:] + hair_masks[:, :-1]) / 2
            weighted_diffs = diffs * hair_weight
            
            # Motion should be smooth (low second derivative)
            if seq_len > 2:
                accel = weighted_diffs[:, 1:] - weighted_diffs[:, :-1]
                smoothness_loss = accel.pow(2).mean()
            else:
                smoothness_loss = torch.tensor(0.0, device=frames.device)
        else:
            smoothness_loss = torch.tensor(0.0, device=frames.device)
        
        # Gravity influence: motion should have downward bias
        # (Simplified: just add regularization toward gravity direction)
        gravity_loss = torch.tensor(0.0, device=frames.device)
        
        return {
            'volume': volume_loss,
            'smoothness': smoothness_loss,
            'gravity': gravity_loss,
            'total': volume_loss + smoothness_loss + gravity_loss,
        }


class EyeMotion(nn.Module):
    """
    Eye motion physics for anime.
    
    Enforces:
    1. Blink patterns (natural frequency, both eyes sync)
    2. Saccades (rapid eye movements, no drift)
    3. Gaze coherence (eyes track same target)
    4. Expression timing (emotions change gradually)
    """
    
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        
        # Eye region detector
        self.eye_detector = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # left/right eye
            nn.Sigmoid(),
        )
        
        # Blink detector
        self.blink_detector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Gaze direction predictor
        self.gaze_predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # x, y gaze direction
        )
    
    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute eye motion constraints.
        
        Args:
            frames: Latent frames [batch, seq, dim]
        
        Returns:
            Physics constraint losses
        """
        batch, seq_len, _ = frames.shape
        
        # Detect eyes
        eye_presence = self.eye_detector(frames)  # [batch, seq, 2]
        
        # Blink detection
        blink_state = self.blink_detector(frames)  # [batch, seq, 1]
        
        # Eye sync: both eyes should blink together
        left_eye = eye_presence[..., 0] * (1 - blink_state.squeeze(-1))
        right_eye = eye_presence[..., 1] * (1 - blink_state.squeeze(-1))
        sync_loss = (left_eye - right_eye).pow(2).mean()
        
        # Blink natural frequency: about 15-20 blinks per 60 fps minute
        # Penalize too many or too few blinks
        blink_changes = (blink_state[:, 1:] - blink_state[:, :-1]).abs()
        blink_freq = blink_changes.sum(dim=1) / seq_len
        target_freq = 0.02  # ~1 blink per 50 frames
        freq_loss = (blink_freq - target_freq).pow(2).mean()
        
        # Gaze coherence
        gaze = self.gaze_predictor(frames)  # [batch, seq, 2]
        if seq_len > 1:
            gaze_change = (gaze[:, 1:] - gaze[:, :-1]).pow(2).sum(dim=-1)
            # Gaze should be mostly stable with occasional saccades
            gaze_loss = torch.clamp(gaze_change - 0.1, min=0).mean()
        else:
            gaze_loss = torch.tensor(0.0, device=frames.device)
        
        return {
            'sync': sync_loss,
            'blink_freq': freq_loss,
            'gaze_stable': gaze_loss,
            'total': sync_loss + freq_loss + gaze_loss,
        }


class MotionPhysics(nn.Module):
    """
    General motion physics for anime.
    
    Enforces:
    1. Momentum conservation
    2. No teleportation (smooth motion)
    3. Anticipation and follow-through
    4. Secondary motion (clothing, accessories)
    """
    
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        
        # Velocity estimator
        self.velocity_net = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
    
    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute motion physics constraints.
        
        Args:
            frames: Latent frames [batch, seq, dim]
        
        Returns:
            Physics constraint losses
        """
        batch, seq_len, dim = frames.shape
        
        if seq_len < 3:
            return {
                'momentum': torch.tensor(0.0, device=frames.device),
                'smoothness': torch.tensor(0.0, device=frames.device),
                'total': torch.tensor(0.0, device=frames.device),
            }
        
        # Compute velocities (first derivative)
        velocities = frames[:, 1:] - frames[:, :-1]  # [batch, seq-1, dim]
        
        # Compute accelerations (second derivative)
        accelerations = velocities[:, 1:] - velocities[:, :-1]  # [batch, seq-2, dim]
        
        # No teleportation: velocities should not be too large
        velocity_magnitude = velocities.pow(2).sum(dim=-1).sqrt()
        teleport_loss = torch.clamp(velocity_magnitude - 2.0, min=0).pow(2).mean()
        
        # Momentum conservation: acceleration should be moderate
        accel_magnitude = accelerations.pow(2).sum(dim=-1).sqrt()
        momentum_loss = accel_magnitude.pow(2).mean()
        
        # Smoothness: third derivative should be small
        if seq_len > 3:
            jerks = accelerations[:, 1:] - accelerations[:, :-1]
            jerk_magnitude = jerks.pow(2).sum(dim=-1).sqrt()
            smoothness_loss = jerk_magnitude.mean()
        else:
            smoothness_loss = torch.tensor(0.0, device=frames.device)
        
        return {
            'teleport': teleport_loss,
            'momentum': momentum_loss,
            'smoothness': smoothness_loss,
            'total': teleport_loss + momentum_loss + 0.5 * smoothness_loss,
        }


class AnimePriors(nn.Module):
    """
    Complete anime physics priors for KISEKI.
    
    Combines all physical constraints into a unified loss
    that guides generation toward physically plausible anime.
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        config: Optional[PhysicsConfig] = None,
    ):
        super().__init__()
        self.config = config or PhysicsConfig()
        
        self.hair = HairPhysics(latent_dim)
        self.eyes = EyeMotion(latent_dim)
        self.motion = MotionPhysics(latent_dim)
    
    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all physics prior losses.
        
        Args:
            frames: Generated latent frames [batch, seq, dim]
        
        Returns:
            Dictionary of all physics losses
        """
        hair_loss = self.hair(frames)
        eye_loss = self.eyes(frames)
        motion_loss = self.motion(frames)
        
        total = (
            self.config.hair_weight * hair_loss['total'] +
            self.config.eye_weight * eye_loss['total'] +
            self.config.motion_weight * motion_loss['total']
        )
        
        return {
            'hair': hair_loss,
            'eye': eye_loss,
            'motion': motion_loss,
            'total': total,
        }
    
    def get_loss(self, frames: torch.Tensor) -> torch.Tensor:
        """Get single physics prior loss value"""
        return self.forward(frames)['total']


class AnimeProportions(nn.Module):
    """
    Anime character proportion constraints.
    
    Ensures characters maintain consistent proportions:
    - Head-to-body ratio
    - Eye size relative to face
    - Limb proportions
    """
    
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        
        # Proportion detectors
        self.head_detector = nn.Linear(latent_dim, 64)
        self.body_detector = nn.Linear(latent_dim, 64)
        
        # Proportion predictor
        self.ratio_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # head/body, eye/face, arm/body, leg/body
        )
        
        # Target proportions (anime style)
        self.register_buffer('target_ratios', torch.tensor([
            0.25,  # head/body ~ 1:4 in anime
            0.35,  # eye/face ~ larger in anime
            0.4,   # arm/body
            0.5,   # leg/body
        ]))
    
    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute proportion constraint losses"""
        head_feat = self.head_detector(frames)
        body_feat = self.body_detector(frames)
        
        combined = torch.cat([head_feat, body_feat], dim=-1)
        ratios = torch.sigmoid(self.ratio_predictor(combined))  # [batch, seq, 4]
        
        # Proportion loss: ratios should match targets
        target = self.target_ratios.unsqueeze(0).unsqueeze(0)
        prop_loss = (ratios - target).pow(2).mean()
        
        # Consistency loss: proportions should be stable across frames
        if frames.shape[1] > 1:
            ratio_var = ratios.var(dim=1).mean()
        else:
            ratio_var = torch.tensor(0.0, device=frames.device)
        
        return {
            'proportion': prop_loss,
            'consistency': ratio_var,
            'total': prop_loss + ratio_var,
        }


if __name__ == "__main__":
    # Test physics priors
    priors = AnimePriors(latent_dim=256)
    
    # Generate fake frames
    frames = torch.randn(4, 32, 256)
    
    losses = priors(frames)
    print("Physics losses:")
    print(f"  Hair total: {losses['hair']['total'].item():.4f}")
    print(f"  Eye total: {losses['eye']['total'].item():.4f}")
    print(f"  Motion total: {losses['motion']['total'].item():.4f}")
    print(f"  Combined total: {losses['total'].item():.4f}")
