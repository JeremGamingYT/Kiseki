"""
KISEKI Training - Self-Improvement and Physics Priors

Advanced training modules:
- SelfImproveLoop: Train on successful generations
- PhysicsPriors: Anime-specific physical constraints
- FlowMatchingLoss: Flow matching objective
"""

from kiseki.training.self_improve import SelfImproveLoop, SyntheticDataBuffer
from kiseki.training.physics_priors import AnimePriors, HairPhysics, EyeMotion
from kiseki.training.flow_matching import FlowMatchingLoss, OptimalTransport

__all__ = [
    "SelfImproveLoop", "SyntheticDataBuffer",
    "AnimePriors", "HairPhysics", "EyeMotion",
    "FlowMatchingLoss", "OptimalTransport",
]
