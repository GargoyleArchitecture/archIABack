"""
CLIP Embedder for EVRAG

Generación de embeddings multimodales usando CLIP.
Permite buscar frames por similitud semántica con texto.
"""

import numpy as np
from pathlib import Path
from typing import Any


class CLIPEmbedder:
    """
    CLIP embedder for multimodal retrieval.
    
    Uses OpenAI's CLIP model to generate embeddings for both
    images (frames) and text (queries/transcript segments).
    
    This enables semantic search across visual and textual content.
    
    Attributes:
        config: Configuration dictionary
        model: Loaded CLIP model
        processor: Image processor
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize CLIP embedder.
        
        Args:
            config: Configuration dictionary (default: EVRAG_CONFIG)
        """
        from .config import EVRAG_CONFIG
        self.config = config or EVRAG_CONFIG
        
        self.model = None
        self.processor = None
        self._model_loaded = False
    
    def _load_model(self):
        """Lazy load CLIP model."""
        if self._model_loaded:
            return
        
        try:
            import clip
            import torch
            from PIL import Image
            
            model_name = self.config["clip_model"]
            print(f"Loading CLIP model: {model_name}")
            
            # Load CLIP model (OpenAI version)
            self.model, self.preprocess = clip.load(model_name, device="cpu")
            self.model.eval()
            
            self._model_loaded = True
            print("  CLIP loaded successfully!")
            
        except ImportError as e:
            print(f"Warning: CLIP not installed ({e}).")
            print("EVRAG will work in text-only mode.")
            self._model_loaded = True
            raise RuntimeError("CLIP not available. Install openai-clip for CLIP support.")
    
    def embed_images(self, image_paths: list[Path | str]) -> np.ndarray:
        """
        Generate embeddings for images using CLIP.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Numpy array of embeddings (n_images x embedding_dim)
        """
        import torch
        from PIL import Image
        
        self._load_model()
        
        embeddings = []
        
        for path in image_paths:
            # Load and preprocess image
            image = Image.open(path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize
                image_features = image_features / image_features.norm(p=2, dim=-1)
            
            embeddings.append(image_features.numpy())
        
        return np.vstack(embeddings)
    
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for texts using CLIP.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings (n_texts x embedding_dim)
        """
        import torch
        import clip
        
        self._load_model()
        
        # Tokenize texts
        text_tokens = clip.tokenize(texts, truncate=True)
        
        # Generate embeddings
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # Normalize
            text_features = text_features / text_features.norm(p=2, dim=-1)
        
        return text_features.numpy()
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        image_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and images.
        
        Args:
            query_embedding: Query embedding (1 x dim) or (dim,)
            image_embeddings: Image embeddings (n x dim)
            
        Returns:
            Similarity scores (n,)
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Cosine similarity (embeddings are already normalized)
        similarities = np.dot(image_embeddings, query_embedding.T).flatten()
        
        return similarities
    
    def find_similar_frames(
        self,
        query_text: str,
        frame_paths: list[Path],
        frame_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[Path, float]]:
        """
        Find most similar frames to a text query.
        
        Args:
            query_text: Text query
            frame_paths: List of frame paths
            frame_embeddings: Pre-computed frame embeddings
            top_k: Number of results to return
            
        Returns:
            List of (frame_path, similarity_score) tuples
        """
        # Embed query
        query_embedding = self.embed_texts([query_text])
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, frame_embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = [
            (frame_paths[i], float(similarities[i]))
            for i in top_indices
        ]
        
        return results
