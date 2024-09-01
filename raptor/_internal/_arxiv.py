"""ArXiv paper helper functions."""

import re
import warnings

from pydantic import BaseModel, Field, AnyHttpUrl

import arxiv
import fitz
import tempfile

from raptor._internal._utils import ensure_ssl_verified


# For downloading the document for demo purposes
class ArXivPaper(BaseModel):
    title: str = Field(..., description="Title of the paper")
    text: str = Field(..., description="Text content of the paper")
    url: AnyHttpUrl = Field(..., description="URL of the paper on ArXiv")


def parse_arxiv_id(url: str) -> str | None:
    """Parses the ArXiv ID from the given URL."""
    arxiv_id = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
    if not arxiv_id:
        warnings.warn(f"Invalid ArXiv URL, skipping url: `{url}`.")
        return None
    return arxiv_id.group(1)


def fetch_papers(urls: str | list[str]) -> list[ArXivPaper]:
    """Fetches and downloads an ArXiv paper from the given URL and returns its ArXivPaper object."""
    client = arxiv.Client()

    if isinstance(urls, str):
        urls = [urls]

    # Ensure SSL certificate is verified
    ensure_ssl_verified(urls[0])  # only check the first URL is enough

    # Extract ArXiv ID from the URL
    arxiv_ids = [
        arxiv_id for url in urls if (arxiv_id := parse_arxiv_id(url)) is not None
    ]

    if not arxiv_ids:
        raise ValueError(f"No valid ArXiv IDs found in the given URLs. urls: `{urls}`")

    # Fetch paper using ArXiv ID
    search_query = arxiv.Search(id_list=arxiv_ids)

    # Download the paper and extract text content
    papers = []
    for paper in client.results(search_query):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = paper.download_pdf(dirpath=temp_dir)
            doc = fitz.open(pdf_path)
            text = "".join([page.get_text() for page in doc])

        papers.append(ArXivPaper(title=paper.title, text=text, url=paper.entry_id))

    return papers
