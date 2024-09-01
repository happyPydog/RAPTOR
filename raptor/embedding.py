import functools

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    @functools.cache
    def __call__(self, text: str) -> np.ndarray:
        return self.model.encode(text)
