from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from tqdm.notebook import tqdm

from raptor.embedding import EmbeddingModel


class Node(BaseModel):
    text: str
    layer: int
    children: set["Node"] | None = Field(default=None)
    embeddings: np.ndarray | None = Field(default=None, repr=False)

    class Config:
        arbitrary_types_allowed = True

    def __hash__(self) -> int:
        return hash((self.text, self.layer))

    def get_children(self) -> set[Node] | None:
        return self.children

    def get_descendants(self) -> set[Node] | None:
        if not self.children:
            return None

        descendants = set(self.children)
        for child in self.children:
            if child_descendants := child.get_descendants():
                descendants.update(child_descendants)

        return descendants

    def is_leaf(self) -> bool:
        return self.children is None


def make_node(
    text: str,
    layer: int,
    embed_model: EmbeddingModel,
    children: set[Node] | None = None,
) -> Node:
    embeddings = embed_model(text)
    return Node(text=text, layer=layer, children=children, embeddings=embeddings)


def make_leaf_nodes(texts: list[str], embed_model: EmbeddingModel) -> list[Node]:
    return [
        make_node(text=text, layer=0, embed_model=embed_model)
        for text in tqdm(texts, desc="Creating lead nodes", total=len(texts))
    ]
