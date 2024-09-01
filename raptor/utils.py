import itertools

from llama_index.core.node_parser import SentenceSplitter


def split_text(texts: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Prepares a list of documents from ArXivPaper."""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sentences = itertools.chain.from_iterable(
        splitter.split_text(text) for text in texts
    )
    return list(sentences)
