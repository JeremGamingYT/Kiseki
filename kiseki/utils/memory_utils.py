"""
Memory Utilities - Efficient GPU Memory Management

Key optimizations:
- DeepSpeed ZeRO for memory-efficient training
- Gradient checkpointing for large models
- Memory tracking and optimization
"""

import gc
from typing import Optional, Dict, Any
from contextlib import contextmanager

import torch
import torch.nn as nn


class MemoryTracker:
    """Track GPU memory usage throughout training."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.history = []
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics in MB"""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'cached': 0, 'max_allocated': 0}
        
        return {
            'allocated': torch.cuda.memory_allocated(self.device) / 1024**2,
            'cached': torch.cuda.memory_reserved(self.device) / 1024**2,
            'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**2,
        }
    
    def log(self, label: str = ''):
        """Log current memory usage"""
        stats = self.get_memory_stats()
        stats['label'] = label
        self.history.append(stats)
        return stats
    
    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def reset_peak_stats(self):
        """Reset peak memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def summary(self) -> str:
        """Get memory usage summary"""
        stats = self.get_memory_stats()
        return (
            f"GPU Memory: {stats['allocated']:.1f}MB allocated, "
            f"{stats['cached']:.1f}MB cached, "
            f"{stats['max_allocated']:.1f}MB peak"
        )


@contextmanager
def memory_efficient_inference():
    """Context manager for memory-efficient inference"""
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            yield


def gradient_checkpointing(model: nn.Module, enable: bool = True):
    """
    Enable gradient checkpointing for a model.
    
    Trades compute for memory: recomputes activations
    during backward pass instead of storing them.
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        if enable:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
    else:
        # Manual implementation for models without built-in support
        for module in model.modules():
            if hasattr(module, 'use_checkpoint'):
                module.use_checkpoint = enable


def setup_deepspeed(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    Setup DeepSpeed for memory-efficient training.
    
    Uses ZeRO Stage 2/3 for optimizer state partitioning
    and CPU offloading for large models.
    """
    default_config = {
        "train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }
    
    if config:
        default_config.update(config)
    
    try:
        import deepspeed
        
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=default_config,
            optimizer=optimizer,
        )
        
        return {
            'engine': model_engine,
            'optimizer': optimizer,
            'config': default_config,
        }
    except ImportError:
        print("DeepSpeed not available, using standard training")
        return {
            'engine': model,
            'optimizer': optimizer or torch.optim.AdamW(model.parameters()),
            'config': default_config,
        }


class ActivationCheckpointing(torch.autograd.Function):
    """Custom activation checkpointing for memory efficiency"""
    
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)
        
        ctx.save_for_backward(*args)
        
        with torch.no_grad():
            outputs = run_function(*args)
        
        return outputs
    
    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Activation checkpointing error")
        
        inputs = ctx.saved_tensors
        
        if ctx.preserve_rng_state:
            rng_devices = ctx.fwd_gpu_devices
            torch.set_rng_state(ctx.fwd_cpu_state)
            set_device_states(rng_devices, ctx.fwd_gpu_states)
        
        with torch.enable_grad():
            outputs = ctx.run_function(*inputs)
        
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        
        torch.autograd.backward(outputs, args)
        
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in inputs
        )
        
        return (None, None) + grads


def get_device_states(*args):
    """Get RNG states for all devices"""
    devices = []
    states = []
    
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            device = arg.device
            if device not in devices:
                devices.append(device)
                states.append(torch.cuda.get_rng_state(device))
    
    return devices, states


def set_device_states(devices, states):
    """Restore RNG states for devices"""
    for device, state in zip(devices, states):
        torch.cuda.set_rng_state(state, device)


def checkpoint(function, *args, preserve_rng_state=True):
    """Apply activation checkpointing to a function"""
    return ActivationCheckpointing.apply(function, preserve_rng_state, *args)


def estimate_model_memory(model: nn.Module) -> Dict[str, float]:
    """
    Estimate memory usage of a model.
    
    Returns memory breakdown in MB.
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    # Estimates for training
    # Gradients: same as parameters
    # Optimizer states: 2x parameters (Adam)
    # Activations: varies, estimate as 4x parameters
    
    param_mb = param_size / 1024**2
    buffer_mb = buffer_size / 1024**2
    grad_mb = param_mb
    optimizer_mb = param_mb * 2
    activation_mb = param_mb * 4  # Rough estimate
    
    return {
        'parameters_mb': param_mb,
        'buffers_mb': buffer_mb,
        'gradients_mb': grad_mb,
        'optimizer_mb': optimizer_mb,
        'activations_mb': activation_mb,
        'total_training_mb': param_mb + buffer_mb + grad_mb + optimizer_mb + activation_mb,
        'inference_mb': param_mb + buffer_mb + activation_mb / 4,
    }


if __name__ == "__main__":
    # Test memory utilities
    tracker = MemoryTracker()
    
    print("Initial:", tracker.summary())
    
    # Create a test model
    model = nn.Linear(1024, 1024)
    if torch.cuda.is_available():
        model = model.cuda()
    
    tracker.log("After model creation")
    print("After model:", tracker.summary())
    
    # Estimate memory
    mem_estimate = estimate_model_memory(model)
    print("\nModel memory estimate:")
    for k, v in mem_estimate.items():
        print(f"  {k}: {v:.2f} MB")
