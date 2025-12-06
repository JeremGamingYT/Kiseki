"""
Flow Matching Loss - The Core Training Objective

Instead of diffusion's noise prediction, we use Flow Matching:
- Learn the velocity field that transforms noise → data
- Simpler, faster convergence
- Better for video generation
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimalTransport:
    """
    Optimal Transport utilities for Flow Matching.
    Computes optimal paths between noise and data distributions.
    """
    
    @staticmethod
    def sample_time(batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample uniform timesteps in [0, 1]"""
        return torch.rand(batch_size, device=device)
    
    @staticmethod
    def interpolate(
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear interpolation between x0 (noise) and x1 (data).
        
        x_t = (1-t) * x0 + t * x1
        """
        t = t.view(-1, *([1] * (x0.ndim - 1)))
        return (1 - t) * x0 + t * x1
    
    @staticmethod
    def get_velocity(
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Target velocity field: derivative of interpolation.
        
        v = x1 - x0
        """
        return x1 - x0
    
    @staticmethod
    def add_noise(
        x: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise at timestep t for training.
        
        Returns:
            noisy: Interpolated sample
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x)
        
        noisy = OptimalTransport.interpolate(noise, x, t)
        return noisy, noise


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching training objective for KISEKI.
    
    Learns to predict the velocity field that transforms
    noise distribution → data distribution.
    
    Advantages over diffusion:
    - Single-step objective (no step-dependent scaling)
    - Faster convergence
    - Deterministic ODE sampling
    """
    
    def __init__(
        self,
        sigma_min: float = 0.001,
        use_ot: bool = True,  # Optimal transport coupling
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.use_ot = use_ot
    
    def forward(
        self,
        model: nn.Module,
        x1: torch.Tensor,  # Target data
        condition: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute Flow Matching loss.
        
        Args:
            model: The velocity prediction model
            x1: Target latent sequences [batch, seq, dim]
            condition: Optional conditioning
            return_components: Whether to return loss components
        
        Returns:
            loss: Flow matching loss value
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # Sample noise (source distribution)
        x0 = torch.randn_like(x1)
        
        # Sample timesteps
        t = OptimalTransport.sample_time(batch_size, device)
        
        # Get interpolated point and target velocity
        x_t = OptimalTransport.interpolate(x0, x1, t)
        v_target = OptimalTransport.get_velocity(x0, x1)
        
        # Predict velocity
        output = model(x_t, t, condition)
        v_pred = output['velocity']
        
        # MSE loss on velocity
        loss = F.mse_loss(v_pred, v_target)
        
        if return_components:
            return {
                'loss': loss,
                'v_pred': v_pred,
                'v_target': v_target,
                'x_t': x_t,
                't': t,
            }
        
        return loss


class ConditionalFlowMatching(FlowMatchingLoss):
    """
    Conditional Flow Matching with classifier-free guidance support.
    """
    
    def __init__(
        self,
        sigma_min: float = 0.001,
        cfg_dropout: float = 0.1,  # Probability of dropping condition
    ):
        super().__init__(sigma_min)
        self.cfg_dropout = cfg_dropout
    
    def forward(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        CFG-aware Flow Matching loss.
        
        Randomly drops conditioning during training to enable
        classifier-free guidance at inference.
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # Sample noise
        x0 = torch.randn_like(x1)
        
        # Sample timesteps
        t = OptimalTransport.sample_time(batch_size, device)
        
        # Get interpolated point and target velocity
        x_t = OptimalTransport.interpolate(x0, x1, t)
        v_target = OptimalTransport.get_velocity(x0, x1)
        
        # CFG: randomly drop condition
        if condition is not None and self.training:
            drop_mask = torch.rand(batch_size, device=device) < self.cfg_dropout
            # Zero out dropped conditions
            condition = condition.clone()
            condition[drop_mask] = 0
        
        # Predict velocity
        output = model(x_t, t, condition)
        v_pred = output['velocity']
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        
        if return_components:
            return {
                'loss': loss,
                'v_pred': v_pred,
                'v_target': v_target,
            }
        
        return loss


class FlowMatchingSampler:
    """
    ODE sampler for trained flow matching models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 50,
        cfg_scale: float = 7.5,
    ):
        self.model = model
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Sample from the model using Euler ODE integration.
        
        Args:
            shape: Output shape (batch, seq, dim)
            condition: Optional conditioning
            device: Device to sample on
        
        Returns:
            Generated samples
        """
        device = device or next(self.model.parameters()).device
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        # Integration timesteps
        dt = 1.0 / self.n_steps
        
        for step in range(self.n_steps):
            t = torch.full((shape[0],), step * dt, device=device)
            
            # Get velocity prediction
            output = self.model(x, t, condition)
            v = output['velocity']
            
            # CFG
            if condition is not None and self.cfg_scale > 1.0:
                output_uncond = self.model(x, t, None)
                v_uncond = output_uncond['velocity']
                v = v_uncond + self.cfg_scale * (v - v_uncond)
            
            # Euler step
            x = x + v * dt
        
        return x
    
    @torch.no_grad()
    def sample_heun(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Sample using Heun's method (2nd order) for better quality.
        """
        device = device or next(self.model.parameters()).device
        
        x = torch.randn(shape, device=device)
        dt = 1.0 / self.n_steps
        
        for step in range(self.n_steps):
            t = torch.full((shape[0],), step * dt, device=device)
            t_next = t + dt
            
            # First velocity evaluation
            v1 = self._get_velocity(x, t, condition)
            
            # Euler prediction
            x_euler = x + v1 * dt
            
            # Second velocity evaluation
            v2 = self._get_velocity(x_euler, t_next, condition)
            
            # Heun update (average of velocities)
            x = x + (v1 + v2) * dt / 2
        
        return x
    
    def _get_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Get velocity with optional CFG"""
        output = self.model(x, t, condition)
        v = output['velocity']
        
        if condition is not None and self.cfg_scale > 1.0:
            output_uncond = self.model(x, t, None)
            v_uncond = output_uncond['velocity']
            v = v_uncond + self.cfg_scale * (v - v_uncond)
        
        return v


if __name__ == "__main__":
    from kiseki.core import MambaFlowGenerator, MambaFlowConfig
    
    # Create model
    config = MambaFlowConfig(d_model=128, n_layers=2, latent_dim=64)
    model = MambaFlowGenerator(config)
    
    # Create loss
    loss_fn = FlowMatchingLoss()
    
    # Test forward
    x = torch.randn(4, 32, 64)
    loss = loss_fn(model, x)
    
    print(f"Flow Matching loss: {loss.item():.4f}")
    
    # Test sampler
    sampler = FlowMatchingSampler(model, n_steps=10)
    samples = sampler.sample((2, 16, 64))
    print(f"Sampled shape: {samples.shape}")
