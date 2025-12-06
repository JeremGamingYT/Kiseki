"""
IVQ Index - Indexed Vector Quantization for KISEKI

Vector database stored on SSD for semantic search:
- "I need Kyoto Animation style eyes" → Returns relevant latents
- Enables style-conditioned generation
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class IndexConfig:
    """Configuration for IVQ Index"""
    dim: int = 512  # Latent dimension
    n_clusters: int = 1024  # Number of IVQ clusters
    n_subquantizers: int = 8  # Product quantization subspaces
    bits_per_code: int = 8  # Bits per subquantizer code
    metric: str = 'cosine'  # 'cosine' or 'l2'


class ProductQuantizer:
    """
    Product Quantization for compact vector storage.
    Splits vectors into subspaces and quantizes each separately.
    """
    
    def __init__(self, dim: int, n_subquantizers: int = 8, bits: int = 8):
        self.dim = dim
        self.n_sq = n_subquantizers
        self.bits = bits
        self.n_codes = 2 ** bits
        self.subdim = dim // n_subquantizers
        
        assert dim % n_subquantizers == 0
        
        # Codebooks: [n_subquantizers, n_codes, subdim]
        self.codebooks = None
        self.trained = False
    
    def train(self, vectors: np.ndarray, n_iter: int = 20):
        """Train codebooks using k-means"""
        n_vectors = len(vectors)
        
        # Initialize codebooks
        self.codebooks = np.zeros((self.n_sq, self.n_codes, self.subdim), dtype=np.float32)
        
        # Train each subquantizer
        for sq in range(self.n_sq):
            # Extract subvectors
            start, end = sq * self.subdim, (sq + 1) * self.subdim
            subvectors = vectors[:, start:end]
            
            # Simple k-means
            # Initialize with random samples
            idx = np.random.choice(n_vectors, self.n_codes, replace=False)
            centroids = subvectors[idx].copy()
            
            for _ in range(n_iter):
                # Assign to nearest centroid
                dists = np.sum((subvectors[:, None] - centroids[None]) ** 2, axis=-1)
                assignments = np.argmin(dists, axis=1)
                
                # Update centroids
                for c in range(self.n_codes):
                    mask = assignments == c
                    if mask.sum() > 0:
                        centroids[c] = subvectors[mask].mean(axis=0)
            
            self.codebooks[sq] = centroids
        
        self.trained = True
    
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors to codes"""
        assert self.trained
        n_vectors = len(vectors)
        codes = np.zeros((n_vectors, self.n_sq), dtype=np.uint8)
        
        for sq in range(self.n_sq):
            start, end = sq * self.subdim, (sq + 1) * self.subdim
            subvectors = vectors[:, start:end]
            dists = np.sum((subvectors[:, None] - self.codebooks[sq][None]) ** 2, axis=-1)
            codes[:, sq] = np.argmin(dists, axis=1)
        
        return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode codes back to vectors"""
        assert self.trained
        n_vectors = len(codes)
        vectors = np.zeros((n_vectors, self.dim), dtype=np.float32)
        
        for sq in range(self.n_sq):
            start, end = sq * self.subdim, (sq + 1) * self.subdim
            vectors[:, start:end] = self.codebooks[sq][codes[:, sq]]
        
        return vectors


class IVQIndex:
    """
    Indexed Vector Quantization database for KISEKI.
    
    Stores 1TB of anime latents in a searchable format:
    - Coarse quantization (IVF clusters) for fast search
    - Product quantization for compact storage
    - Memory-mapped for NVMe streaming
    """
    
    def __init__(self, config: Optional[IndexConfig] = None):
        self.config = config or IndexConfig()
        
        # Coarse quantizer (IVF centroids)
        self.centroids = None  # [n_clusters, dim]
        
        # Product quantizer for storage
        self.pq = ProductQuantizer(
            self.config.dim,
            self.config.n_subquantizers,
            self.config.bits_per_code,
        )
        
        # Inverted lists
        self.invlists = {}  # cluster_id -> (ids, codes)
        
        self.n_vectors = 0
        self.trained = False
    
    def train(self, vectors: np.ndarray, verbose: bool = True):
        """Train the index on sample vectors"""
        vectors = vectors.astype(np.float32)
        n_train = len(vectors)
        
        if verbose:
            print(f"Training IVQ index on {n_train} vectors...")
        
        # Normalize for cosine similarity
        if self.config.metric == 'cosine':
            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        
        # Train coarse quantizer (k-means for centroids)
        n_iter = 20
        idx = np.random.choice(n_train, min(self.config.n_clusters, n_train), replace=False)
        self.centroids = vectors[idx].copy()
        
        for i in range(n_iter):
            # Assign
            dists = self._compute_distances(vectors, self.centroids)
            assignments = np.argmin(dists, axis=1)
            
            # Update
            for c in range(self.config.n_clusters):
                mask = assignments == c
                if mask.sum() > 0:
                    self.centroids[c] = vectors[mask].mean(axis=0)
                    if self.config.metric == 'cosine':
                        self.centroids[c] /= np.linalg.norm(self.centroids[c]) + 1e-8
        
        # Train product quantizer on residuals
        dists = self._compute_distances(vectors, self.centroids)
        assignments = np.argmin(dists, axis=1)
        residuals = vectors - self.centroids[assignments]
        self.pq.train(residuals)
        
        self.trained = True
        if verbose:
            print("Training complete!")
    
    def _compute_distances(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute pairwise distances"""
        if self.config.metric == 'cosine':
            return 1 - np.dot(a, b.T)
        else:
            return np.sum((a[:, None] - b[None]) ** 2, axis=-1)
    
    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        """Add vectors to the index"""
        assert self.trained
        vectors = vectors.astype(np.float32)
        n_add = len(vectors)
        
        if ids is None:
            ids = np.arange(self.n_vectors, self.n_vectors + n_add)
        
        # Normalize
        if self.config.metric == 'cosine':
            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        
        # Assign to clusters
        dists = self._compute_distances(vectors, self.centroids)
        assignments = np.argmin(dists, axis=1)
        
        # Compute residuals and encode
        residuals = vectors - self.centroids[assignments]
        codes = self.pq.encode(residuals)
        
        # Add to inverted lists
        for i, cluster_id in enumerate(assignments):
            if cluster_id not in self.invlists:
                self.invlists[cluster_id] = ([], [])
            self.invlists[cluster_id][0].append(ids[i])
            self.invlists[cluster_id][1].append(codes[i])
        
        self.n_vectors += n_add
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        n_probe: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector [dim] or [n_queries, dim]
            k: Number of neighbors to return
            n_probe: Number of clusters to search
        
        Returns:
            distances: [n_queries, k]
            ids: [n_queries, k]
        """
        assert self.trained
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype(np.float32)
        n_queries = len(query)
        
        # Normalize
        if self.config.metric == 'cosine':
            query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
        
        all_distances = []
        all_ids = []
        
        for q in query:
            # Find nearest clusters
            cluster_dists = self._compute_distances(q.reshape(1, -1), self.centroids)[0]
            top_clusters = np.argsort(cluster_dists)[:n_probe]
            
            candidates_ids = []
            candidates_dists = []
            
            for cluster_id in top_clusters:
                if cluster_id not in self.invlists:
                    continue
                
                ids_list, codes_list = self.invlists[cluster_id]
                if not ids_list:
                    continue
                
                # Decode and compute distances
                codes = np.array(codes_list)
                residuals = self.pq.decode(codes)
                reconstructed = residuals + self.centroids[cluster_id]
                
                dists = self._compute_distances(q.reshape(1, -1), reconstructed)[0]
                
                candidates_ids.extend(ids_list)
                candidates_dists.extend(dists)
            
            # Get top-k
            if candidates_ids:
                candidates_ids = np.array(candidates_ids)
                candidates_dists = np.array(candidates_dists)
                top_k = np.argsort(candidates_dists)[:k]
                all_ids.append(candidates_ids[top_k])
                all_distances.append(candidates_dists[top_k])
            else:
                all_ids.append(np.full(k, -1))
                all_distances.append(np.full(k, np.inf))
        
        return np.array(all_distances), np.array(all_ids)
    
    def save(self, path: str):
        """Save index to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(path / "config.json", 'w') as f:
            json.dump(asdict(self.config), f)
        
        # Save centroids
        np.save(path / "centroids.npy", self.centroids)
        
        # Save PQ codebooks
        np.save(path / "pq_codebooks.npy", self.pq.codebooks)
        
        # Save inverted lists
        invlists_data = {}
        for k, (ids, codes) in self.invlists.items():
            invlists_data[str(k)] = {
                'ids': ids,
                'codes': [c.tolist() for c in codes],
            }
        with open(path / "invlists.json", 'w') as f:
            json.dump(invlists_data, f)
        
        # Save metadata
        metadata = {'n_vectors': self.n_vectors}
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    def load(self, path: str):
        """Load index from disk"""
        path = Path(path)
        
        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)
            self.config = IndexConfig(**config_dict)
        
        # Load centroids
        self.centroids = np.load(path / "centroids.npy")
        
        # Load PQ
        self.pq = ProductQuantizer(
            self.config.dim,
            self.config.n_subquantizers,
            self.config.bits_per_code,
        )
        self.pq.codebooks = np.load(path / "pq_codebooks.npy")
        self.pq.trained = True
        
        # Load inverted lists
        with open(path / "invlists.json") as f:
            invlists_data = json.load(f)
        
        self.invlists = {}
        for k, v in invlists_data.items():
            self.invlists[int(k)] = (
                v['ids'],
                [np.array(c, dtype=np.uint8) for c in v['codes']]
            )
        
        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
            self.n_vectors = metadata['n_vectors']
        
        self.trained = True


class SemanticIndex(IVQIndex):
    """
    Extended IVQ with semantic search capabilities.
    
    Enables queries like:
    - "Kyoto Animation style eyes"
    - "Dynamic action scene"
    - "Sakura petals falling"
    """
    
    def __init__(self, config: Optional[IndexConfig] = None, text_encoder=None):
        super().__init__(config)
        self.text_encoder = text_encoder
        self.semantic_labels = {}  # id -> text description
    
    def add_with_labels(
        self,
        vectors: np.ndarray,
        labels: List[str],
        ids: Optional[np.ndarray] = None,
    ):
        """Add vectors with semantic labels"""
        if ids is None:
            ids = np.arange(self.n_vectors, self.n_vectors + len(vectors))
        
        for i, label in zip(ids, labels):
            self.semantic_labels[int(i)] = label
        
        self.add(vectors, ids)
    
    def search_by_text(self, text: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Search using text query"""
        if self.text_encoder is None:
            raise ValueError("Text encoder required for semantic search")
        
        # Encode text to vector
        with torch.no_grad():
            query = self.text_encoder.encode(text)
            if isinstance(query, torch.Tensor):
                query = query.cpu().numpy()
        
        distances, ids = self.search(query, k)
        
        # Get labels
        labels = [self.semantic_labels.get(int(i), "") for i in ids[0]]
        
        return distances, ids, labels


if __name__ == "__main__":
    # Test IVQ Index
    config = IndexConfig(dim=128, n_clusters=16)
    index = IVQIndex(config)
    
    # Generate random training data
    n_train = 1000
    train_vectors = np.random.randn(n_train, 128).astype(np.float32)
    
    # Train
    index.train(train_vectors)
    
    # Add vectors
    n_add = 5000
    add_vectors = np.random.randn(n_add, 128).astype(np.float32)
    index.add(add_vectors)
    
    print(f"Index contains {index.n_vectors} vectors")
    
    # Search
    query = np.random.randn(1, 128).astype(np.float32)
    distances, ids = index.search(query, k=5)
    
    print(f"Top 5 results: {ids[0]}")
    print(f"Distances: {distances[0]}")
