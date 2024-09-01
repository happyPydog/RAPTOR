from openai import OpenAI

from raptor.node import Node


def summarize_cluster_texts(
    llm: OpenAI,
    nodes: list[Node],
    model_name: str = "gpt-4o",
) -> str:
    """Summarize the cluster of nodes using the given language model."""
    # Extract the context from the nodes
    context = ""
    for node in nodes:
        context += " ".join(node.text.splitlines())
        context += "\n\n"

    # Generate a summary using the language model
    nodes_prompt = "Write a summary of the following, including as many key details as possible: {context}:".format(
        context=context
    )
    response = llm.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": nodes_prompt},
        ],
    )

    return response.choices[0].message.content
