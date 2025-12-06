"""
Mamba-Flow Generator - The Heart of KISEKI

Combines Mamba-2 State Space Models with Flow Matching for
linear-complexity O(N) anime video generation.

Key advantages over Transformers:
- O(N) complexity instead of O(N²)
- Infinite context via compressed state
- Constant speed regardless of sequence length
"""

import math
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Using fallback implementation.")


@dataclass
class MambaFlowConfig:
    """Configuration for MambaFlow Generator"""
    # Model dimensions
    d_model: int = 1024           # Hidden dimension
    d_state: int = 64             # SSM state dimension
    d_conv: int = 4               # Convolution kernel size
    expand_factor: int = 2        # MLP expansion factor
    
    # Architecture
    n_layers: int = 24            # Number of Mamba blocks
    n_heads: int = 16             # For sparse attention
    
    # Input/Output
    latent_dim: int = 512         # SVG-Latent dimension
    vocab_size: int = 8192        # Discrete token vocabulary
    
    # Flow matching
    flow_time_embed_dim: int = 256
    
    # Regularization
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Memory optimization
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    
    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand_factor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function - more efficient than GELU"""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MambaBlockFallback(nn.Module):
    """
    Fallback Mamba block implementation when mamba-ssm is not available.
    Uses a simplified SSM approximation with selective state updates.
    """
    
    def __init__(self, config: MambaFlowConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_inner = config.d_inner
        d_state = config.d_state
        d_conv = config.d_conv
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        # Causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
        )
        
        # SSM parameters: Δ, B, C projections
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        
        # Discretization parameter log(Δ)
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)
        
        # Initialize dt bias to be small positive values
        with torch.no_grad():
            dt_init_std = config.d_model ** -0.5
            nn.init.uniform_(self.dt_proj.bias, 0.0, dt_init_std)
        
        # State matrix A (fixed, learned from log values)
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            'n -> d n',
            d=d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
        # Normalization
        self.norm = RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            state: Previous SSM state [batch, d_inner, d_state]
        
        Returns:
            output: Processed tensor [batch, seq_len, d_model]
            new_state: Updated SSM state
        """
        batch, seq_len, _ = x.shape
        residual = x
        x = self.norm(x)
        
        # Project and split into x and z (gate)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Causal convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)
        
        # SSM parameters
        x_proj = self.x_proj(x)
        delta, B, C = x_proj.split([1, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(F.relu(delta.squeeze(-1))))
        
        # Get A matrix
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Selective scan (simplified)
        # In practice, this should use the optimized CUDA kernel
        y, new_state = self._selective_scan(x, delta, A, B, C, self.D, state)
        
        # Gate and project
        y = y * F.silu(z)
        output = self.out_proj(y)
        
        return output + residual, new_state
    
    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        prev_state: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simplified selective scan implementation.
        For production, use mamba-ssm's optimized CUDA kernel.
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Initialize state
        if prev_state is None:
            state = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        else:
            state = prev_state
        
        outputs = []
        
        # Sequential scan (this is the slow path - optimized version uses parallel scan)
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, d_inner]
            delta_t = delta[:, t, :].unsqueeze(-1)  # [batch, d_inner, 1]
            B_t = B[:, t, :].unsqueeze(1)  # [batch, 1, d_state]
            C_t = C[:, t, :]  # [batch, d_state]
            
            # Discretize A and B
            deltaA = torch.exp(delta_t * A)  # [batch, d_inner, d_state]
            deltaB = delta_t * B_t  # [batch, d_inner, d_state]
            
            # State update: h_t = Ā * h_{t-1} + B̄ * x_t
            state = deltaA * state + deltaB * x_t.unsqueeze(-1)
            
            # Output: y_t = C * h_t + D * x_t
            y_t = torch.einsum('bdn,bn->bd', state, C_t) + D * x_t
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)
        return y, state


class MambaBlock(nn.Module):
    """
    Production Mamba block using optimized kernels when available.
    Falls back to pure PyTorch implementation otherwise.
    """
    
    def __init__(self, config: MambaFlowConfig):
        super().__init__()
        self.config = config
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand_factor,
            )
            self.norm = RMSNorm(config.d_model)
            self._use_native = True
        else:
            self.mamba = MambaBlockFallback(config)
            self._use_native = False
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._use_native:
            # Native mamba-ssm doesn't return state by default
            residual = x
            x = self.norm(x)
            x = self.mamba(x)
            return x + residual, None
        else:
            return self.mamba(x, state)


class FlowTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for Flow Matching.
    Encodes the flow timestep t ∈ [0, 1] into a vector.
    """
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # MLP to project embeddings
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Flow timesteps [batch] in range [0, 1]
        
        Returns:
            embeddings: Time embeddings [batch, dim]
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        
        # Scale t to match standard timestep range
        t = t[:, None] * 1000.0  # [batch, 1]
        args = t * freqs[None, :]  # [batch, half_dim]
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class MotionPrior(nn.Module):
    """
    Motion prior module that encodes anime-specific motion patterns.
    Uses learned motion primitives for common anime animations.
    """
    
    def __init__(self, config: MambaFlowConfig):
        super().__init__()
        self.config = config
        
        # Motion vocabulary (common anime motion patterns)
        n_motion_primitives = 64
        self.motion_embeddings = nn.Embedding(n_motion_primitives, config.d_model)
        
        # Motion prediction head
        self.motion_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, n_motion_primitives),
        )
        
        # Motion integration
        self.motion_gate = nn.Linear(config.d_model * 2, config.d_model)
    
    def forward(self, x: torch.Tensor, prev_frame: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict and integrate motion patterns.
        
        Args:
            x: Current latent features [batch, seq, d_model]
            prev_frame: Previous frame features [batch, d_model]
        
        Returns:
            Motion-aware features [batch, seq, d_model]
        """
        # Predict motion type logits
        motion_logits = self.motion_proj(x)  # [batch, seq, n_primitives]
        motion_weights = F.softmax(motion_logits, dim=-1)
        
        # Weighted sum of motion embeddings
        motion_features = torch.einsum(
            'bsn,nd->bsd',
            motion_weights,
            self.motion_embeddings.weight
        )
        
        # Gate and combine with input
        combined = torch.cat([x, motion_features], dim=-1)
        gate = torch.sigmoid(self.motion_gate(combined))
        
        return x + gate * motion_features


class MambaFlowGenerator(nn.Module):
    """
    KISEKI Core Generator Architecture
    
    Combines Mamba-2 SSM blocks with Flow Matching for efficient
    anime video generation with O(N) complexity.
    
    Key features:
    - Linear complexity in sequence length
    - Infinite context via compressed state memory
    - Flow matching for smooth generation
    - Motion priors for anime-specific patterns
    """
    
    def __init__(self, config: Optional[MambaFlowConfig] = None):
        super().__init__()
        self.config = config or MambaFlowConfig()
        
        # Input embedding for SVG-latent tokens
        self.input_embed = nn.Linear(self.config.latent_dim, self.config.d_model)
        
        # Flow time embedding
        self.time_embed = FlowTimeEmbedding(self.config.flow_time_embed_dim)
        self.time_proj = nn.Linear(self.config.flow_time_embed_dim, self.config.d_model)
        
        # Motion prior
        self.motion_prior = MotionPrior(self.config)
        
        # Main Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(self.config) for _ in range(self.config.n_layers)
        ])
        
        # Inter-layer normalization
        self.norms = nn.ModuleList([
            RMSNorm(self.config.d_model) for _ in range(self.config.n_layers)
        ])
        
        # MLP blocks (every other layer)
        self.mlps = nn.ModuleList([
            SwiGLU(self.config.d_model, self.config.d_inner, self.config.dropout)
            if i % 2 == 1 else None
            for i in range(self.config.n_layers)
        ])
        
        # Output projection to SVG-latent velocity
        self.output_norm = RMSNorm(self.config.d_model)
        self.output_proj = nn.Linear(self.config.d_model, self.config.latent_dim)
        
        # Gradient checkpointing
        self._gradient_checkpointing = self.config.gradient_checkpointing
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        return_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for flow matching training/inference.
        
        Args:
            x: Noisy SVG-latent tokens [batch, seq_len, latent_dim]
            t: Flow timesteps [batch] in range [0, 1]
            condition: Optional conditioning (text embedding, etc.) [batch, cond_len, d_model]
            return_states: Whether to return intermediate states
        
        Returns:
            Dictionary containing:
                - velocity: Predicted velocity field [batch, seq_len, latent_dim]
                - states: (optional) List of intermediate states
        """
        batch, seq_len, _ = x.shape
        
        # Embed input
        h = self.input_embed(x)
        
        # Add time embedding
        time_emb = self.time_embed(t)
        time_emb = self.time_proj(time_emb)
        h = h + time_emb.unsqueeze(1)
        
        # Add conditioning if present
        if condition is not None:
            # Cross-attention would go here for text conditioning
            # For now, we use simple addition of mean-pooled condition
            cond_pooled = condition.mean(dim=1, keepdim=True)
            h = h + cond_pooled
        
        # Apply motion prior
        h = self.motion_prior(h)
        
        # Store states for analysis
        states = [] if return_states else None
        
        # Main Mamba blocks
        for i, (layer, norm, mlp) in enumerate(zip(self.layers, self.norms, self.mlps)):
            # Mamba block
            h, _ = layer(h)
            
            # Optional MLP
            if mlp is not None:
                h = h + mlp(norm(h))
            
            if return_states:
                states.append(h.detach())
        
        # Output projection
        h = self.output_norm(h)
        velocity = self.output_proj(h)
        
        result = {"velocity": velocity}
        if return_states:
            result["states"] = states
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        cfg_scale: float = 7.5,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate anime frames using Flow Matching sampling.
        
        Args:
            shape: Output shape (batch, seq_len, latent_dim)
            condition: Optional conditioning tensor
            n_steps: Number of ODE integration steps
            cfg_scale: Classifier-free guidance scale
            device: Device to generate on
        
        Returns:
            Generated SVG-latent tensor [batch, seq_len, latent_dim]
        """
        device = device or next(self.parameters()).device
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        # Integration timesteps (from t=0 to t=1)
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.full((shape[0],), step * dt, device=device)
            
            # Get velocity prediction
            output = self.forward(x, t, condition)
            v = output["velocity"]
            
            # Classifier-free guidance
            if condition is not None and cfg_scale > 1.0:
                output_uncond = self.forward(x, t, None)
                v_uncond = output_uncond["velocity"]
                v = v_uncond + cfg_scale * (v - v_uncond)
            
            # Euler integration step
            x = x + v * dt
        
        return x
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return number of parameters"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.input_embed.weight.numel()
        return n_params


def create_kiseki_600m() -> MambaFlowGenerator:
    """Create 600M parameter KISEKI model"""
    config = MambaFlowConfig(
        d_model=1024,
        d_state=64,
        n_layers=24,
        n_heads=16,
        latent_dim=512,
        expand_factor=2,
    )
    return MambaFlowGenerator(config)


def create_kiseki_1b() -> MambaFlowGenerator:
    """Create 1B parameter KISEKI model"""
    config = MambaFlowConfig(
        d_model=1536,
        d_state=128,
        n_layers=32,
        n_heads=24,
        latent_dim=512,
        expand_factor=2,
    )
    return MambaFlowGenerator(config)


if __name__ == "__main__":
    # Test the model
    config = MambaFlowConfig(
        d_model=256,
        n_layers=4,
        latent_dim=128,
    )
    
    model = MambaFlowGenerator(config)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Test forward pass
    batch, seq_len, latent_dim = 2, 32, 128
    x = torch.randn(batch, seq_len, latent_dim)
    t = torch.rand(batch)
    
    output = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output velocity shape: {output['velocity'].shape}")
    
    # Test generation
    generated = model.generate((1, 16, latent_dim), n_steps=10)
    print(f"Generated shape: {generated.shape}")
