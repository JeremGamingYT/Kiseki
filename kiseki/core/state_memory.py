"""
State Memory Module - Infinite Context for KISEKI
Implements compressed state representations for O(N) complexity.
"""

import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class StateMemoryConfig:
    """Configuration for state memory"""
    d_model: int = 1024
    d_state: int = 256
    n_memory_slots: int = 16
    update_rate: float = 0.1
    use_gating: bool = True


class StateSpaceLayer(nn.Module):
    """SSM layer for sequence compression into fixed-size state."""
    
    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.dt_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.Softplus())
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, prev_state: Optional[torch.Tensor] = None):
        batch, seq_len, _ = x.shape
        state = prev_state if prev_state is not None else torch.zeros(
            batch, self.d_model, self.d_state, device=x.device, dtype=x.dtype)
        
        dt = self.dt_proj(x)
        A = -torch.exp(self.A_log)
        outputs = []
        
        for t in range(seq_len):
            x_t, dt_t = x[:, t, :], dt[:, t, :]
            dA = torch.exp(dt_t.unsqueeze(-1) * A)
            B_t = self.B_proj(x_t)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            state = dA * state + dB * x_t.unsqueeze(-1)
            y_t = self.C_proj(state.mean(dim=1)) + self.D * x_t
            outputs.append(y_t)
        
        return self.norm(torch.stack(outputs, dim=1)), state


class PersistentMemory(nn.Module):
    """Memory bank for long-term context."""
    
    def __init__(self, config: StateMemoryConfig):
        super().__init__()
        self.config = config
        self.memory = nn.Parameter(torch.randn(config.n_memory_slots, config.d_state) * 0.02)
        self.query_proj = nn.Linear(config.d_model, config.d_state)
        self.key_proj = nn.Linear(config.d_state, config.d_state)
        self.value_proj = nn.Linear(config.d_state, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
    
    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        memory = repeat(self.memory, 's d -> b s d', b=batch)
        queries, keys = self.query_proj(x), self.key_proj(memory)
        attn = F.softmax(torch.einsum('bld,bsd->bls', queries, keys) / math.sqrt(self.config.d_state), dim=-1)
        values = self.value_proj(memory)
        return x + self.out_proj(torch.einsum('bls,bsd->bld', attn, values)), attn


class StateMemory(nn.Module):
    """Complete State Memory System combining SSM and persistent memory."""
    
    def __init__(self, config: Optional[StateMemoryConfig] = None):
        super().__init__()
        self.config = config or StateMemoryConfig()
        self.ssm = StateSpaceLayer(self.config.d_model, self.config.d_state)
        self.persistent_memory = PersistentMemory(self.config)
        self.norm = nn.LayerNorm(self.config.d_model)
    
    def forward(self, x: torch.Tensor, prev_state: Optional[torch.Tensor] = None):
        ssm_out, new_state = self.ssm(x, prev_state)
        memory_out, memory_attn = self.persistent_memory(ssm_out)
        return {'output': self.norm(memory_out), 'ssm_state': new_state, 'memory_attention': memory_attn}
