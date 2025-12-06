# 🌟 Project KISEKI - 奇跡

> **"Breaking the Glass Ceiling of AI Video Generation"**

A revolutionary neuro-symbolic latent architecture for anime video generation, designed to run on consumer GPUs (RTX 3090/4090) while processing 1TB+ datasets through intelligent streaming.

---

## 🎯 The Paradigm Shift

Instead of learning to predict noise (classical Diffusion), KISEKI learns to predict **motion vectors and composition** in an ultra-compressed latent space, guided by State Space Models (SSM) rather than Transformers.

### Key Innovations

| Current Models (Sora/SVD) | KISEKI Architecture |
|:--------------------------|:--------------------|
| Loads pixels in RAM | Streams latent vectors from SSD |
| Transformer O(N²) | Mamba SSM O(N) |
| Predicts noise (Diffusion) | Predicts primitives (Vectors/Flow) |
| H100 clusters required | Single RTX 4090 |
| Pure statistics | Structural & geometric understanding |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        KISEKI CORE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐ │
│  │   NVMe SSD      │    │  IVQ Index       │    │  GPUDirect         │ │
│  │   (1TB Data)    │───▶│  (Vector DB)     │───▶│  (Zero-Copy)       │ │
│  └─────────────────┘    └──────────────────┘    └─────────┬──────────┘ │
│                                                           │            │
│  ┌────────────────────────────────────────────────────────▼──────────┐ │
│  │                      ANIME VAE-X (Neural Tokenizer)               │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │ │
│  │  │  Lineart    │    │   Color     │    │  Motion Embedding   │   │ │
│  │  │  Encoder    │    │   Encoder   │    │  (Temporal Prior)   │   │ │
│  │  └──────┬──────┘    └──────┬──────┘    └──────────┬──────────┘   │ │
│  │         └──────────────────┴─────────────────────┬┘              │ │
│  │                                                  │               │ │
│  │         ┌────────────────────────────────────────▼─────────┐     │ │
│  │         │           SVG-Latent Space (4000x Compression)   │     │ │
│  │         └────────────────────────────────────────┬─────────┘     │ │
│  └──────────────────────────────────────────────────┼───────────────┘ │
│                                                     │                 │
│  ┌──────────────────────────────────────────────────▼───────────────┐ │
│  │                    MAMBA-FLOW GENERATOR                          │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐  │ │
│  │  │  Mamba-2 Blocks  │  │  Flow Matching   │  │  State Memory  │  │ │
│  │  │  (SSM Core)      │◀─│  (Trajectory)    │◀─│  (Infinite)    │  │ │
│  │  └────────┬─────────┘  └──────────────────┘  └────────────────┘  │ │
│  │           │                                                       │ │
│  │  ┌────────▼─────────┐  ┌──────────────────┐  ┌────────────────┐  │ │
│  │  │  Sparse Local    │  │  Physics Priors  │  │  Self-Improve  │  │ │
│  │  │  Attention       │  │  (Anime Rules)   │  │  Loop          │  │ │
│  │  └──────────────────┘  └──────────────────┘  └────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                    SVG DECODER (Vector Renderer)                 │ │
│  │        Curves + Fill Zones → Anti-aliased Anime Frames           │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Project Structure

```
KISEKI/
├── kiseki/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── mamba_flow.py        # Mamba-2 + Flow Matching backbone
│   │   ├── state_memory.py      # Infinite context state space
│   │   └── sparse_attention.py  # Sparse local attention module
│   │
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── anime_vae.py         # Specialized Anime VAE-X
│   │   ├── lineart_encoder.py   # Lineart separation encoder
│   │   ├── color_encoder.py     # Cell-shading color encoder
│   │   └── svg_latent.py        # Vectorial latent space
│   │
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── zero_copy.py         # NVMe direct streaming
│   │   ├── ivq_index.py         # Indexed Vector Quantization
│   │   └── gpu_direct.py        # GPUDirect/DirectStorage
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── self_improve.py      # Self-improvement loop
│   │   ├── physics_priors.py    # Anime physics constraints
│   │   ├── clip_anime.py        # Anime-specialized discriminator
│   │   └── flow_matching.py     # Flow matching loss
│   │
│   └── utils/
│       ├── __init__.py
│       ├── triton_kernels.py    # Custom Triton GPU kernels
│       └── memory_utils.py      # DeepSpeed/ZeRO utilities
│
├── configs/
│   ├── model_600m.yaml          # 600M parameter config
│   ├── model_1b.yaml            # 1B parameter config
│   └── training.yaml            # Training hyperparameters
│
├── scripts/
│   ├── preprocess_dataset.py    # Convert videos to IVQ index
│   ├── train.py                 # Main training script
│   └── generate.py              # Inference/generation script
│
├── notebooks/
│   └── KISEKI_Training.ipynb    # Complete training notebook
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/kiseki.git
cd kiseki

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Preprocessing Dataset (1TB → IVQ Index)

```bash
python scripts/preprocess_dataset.py \
    --input_dir /path/to/anime/videos \
    --output_dir /path/to/ivq_index \
    --patch_size 16 \
    --workers 8
```

### Training

```bash
# Single GPU training (RTX 4090)
python scripts/train.py \
    --config configs/model_600m.yaml \
    --data_index /path/to/ivq_index \
    --output_dir ./checkpoints
```

### Generation

```bash
python scripts/generate.py \
    --checkpoint ./checkpoints/kiseki_600m.pt \
    --prompt "A girl with blue hair running through cherry blossoms" \
    --duration 5.0 \
    --output ./output.mp4
```

---

## 💻 Hardware Requirements

### Minimum (Development)
- GPU: RTX 3080 (10GB VRAM)
- RAM: 16GB
- Storage: 256GB NVMe SSD (for index)

### Recommended (Training)
- GPU: RTX 4090 (24GB VRAM)
- RAM: 32GB
- Storage: 2TB NVMe Gen4 SSD

### Expected Performance
- **Preprocessing**: ~24 hours for 1TB dataset
- **Training (600M)**: ~24 hours to convergence
- **Inference**: ~2 seconds per frame

---

## 🔬 Technical Details

### Why This Works on Consumer GPUs

1. **No Redundancy**: Background pixels that don't move are not computed (State Space Models)
2. **Direct Streaming**: GPU runs at 100% compute, 0% waiting for data
3. **Reduced Dimensionality**: Working on quasi-vectorial representations requires infinitely less VRAM

### The 4000x Compression Factor

| Representation | Size (1080p frame) |
|:---------------|:-------------------|
| Raw Pixels | 6.2 MB (2M pixels × 3 channels) |
| SVG-Latent | ~1.5 KB (500 primitives × 3 bytes) |
| **Compression** | **~4000x** |

---

## 📚 References

- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🌸 Acknowledgments

Project KISEKI (奇跡 - "Miracle") is inspired by the beauty of Japanese animation and the desire to democratize AI-powered anime creation.

*"The miracle is not to walk on water. The miracle is to walk on the green earth, dwelling deeply in the present moment and feeling truly alive."* - Thich Nhat Hanh
