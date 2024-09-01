import itertools
from collections import defaultdict

import tiktoken
from openai import OpenAI
from pydantic import BaseModel, Field, computed_field
from tqdm import tqdm

from raptor.embedding import EmbeddingModel
from raptor.clustering import run_clustering, RAPTORClustering
from raptor.node import Node, make_node, make_leaf_nodes
from raptor.summary import summarize_cluster_texts
from raptor.logging import get_logger

logger = get_logger(__name__)


class Tree(BaseModel):
    nodes_map: defaultdict[int, list[Node]] = Field(
        default_factory=lambda: defaultdict(list),
        description="A mapping of layer number to the nodes.",
    )

    @computed_field
    @property
    def leaf_nodes(self) -> list[Node] | None:
        return self.nodes_map[0] if self.nodes_map else None

    @computed_field
    @property
    def root_nodes(self) -> list[Node] | None:
        max_layers = max(self.nodes_map.keys())
        return self.nodes_map[max_layers] if self.nodes_map else None

    @computed_field
    @property
    def total_nodes_count(self) -> int:
        return len(self.traverse_tree())

    def retrieve_layer(self, layer: int) -> list[Node]:
        return self.nodes_map[layer]

    def traverse_tree(self) -> list[Node]:
        return list(itertools.chain.from_iterable(self.nodes_map.values()))

    def traverse_node(self, node: Node) -> list[Node]:
        return [node] + [
            child for child in node.children for child in self.traverse_node(child)
        ]

    def traverse_nodes(self, nodes: list[Node]) -> list[Node]:
        return list(
            itertools.chain.from_iterable(self.traverse_node(node) for node in nodes)
        )


def make_tree(
    texts: list[str],
    *,
    embed_model: EmbeddingModel,
    llm: OpenAI,
    tokenizer: tiktoken,
    max_depth: int,
    max_tokens_per_cluster: int,
    max_cluster_size: int,
    reduction_component_size: int,
    random_state: int,
    threshold: float,
) -> Tree:
    # Initialize objects
    tree = Tree()
    model = RAPTORClustering(
        reduction_component_size=reduction_component_size,
        threshold=threshold,
        random_state=random_state,
        max_cluster_size=max_cluster_size,
    )

    # Construct the tree layer by layer
    for layer in tqdm(range(max_depth), total=max_depth):
        logger.debug(f"Building layer {layer}...")
        if layer == 0:
            leaf_nodes = make_leaf_nodes(texts, embed_model)
            tree.nodes_map[layer] = leaf_nodes
            logger.debug(f"Number of leaf nodes in layer {layer}: {len(leaf_nodes)}")
            continue

        # get last layer nodes
        last_nodes = tree.retrieve_layer(layer - 1)

        # run clustering
        clusters = run_clustering(
            last_nodes,
            model=model,
            tokenizer=tokenizer,
            max_tokens_per_cluster=max_tokens_per_cluster,
        )

        # compose nodes based on the clusters
        logger.debug(f"Number of new nodes in layer {layer}: {len(clusters)}")
        curr_layer_nodes: list[Node] = []
        for cluster in tqdm(clusters, desc="Creating nodes", total=len(clusters)):
            summary = summarize_cluster_texts(llm, cluster)
            nodes = make_node(
                text=summary,
                layer=layer,
                embed_model=embed_model,
                children=set(cluster),
            )
            curr_layer_nodes.append(nodes)

        # update the tree
        tree.nodes_map[layer] = curr_layer_nodes

    return tree
