"""ArXiv paper helper functions."""

from __future__ import annotations
import fitz
import tempfile
import re


from more_itertools import flatten
from more_itertools import unique_everseen
from typing import cast
from collections.abc import Iterable
from pathlib import Path

import ssl

from arxiv import Client as _ArxivClient
from arxiv import Result as ArxivResult
from arxiv import Search as ArxivSearch
from pydantic import BaseModel
from pydantic import AnyHttpUrl


class Paper(BaseModel):
    title: str
    text: str
    url: AnyHttpUrl
    references: list[Paper] | None = None

    def flatten(self) -> list[Paper]:
        papers = [self]
        if self.references:
            for paper in self.references:
                papers.extend(paper.flatten())
        return papers


class ArxivClient:

    def __init__(self) -> None:
        self.client = _ArxivClient()

        self.ensure_ssl_verified()

    def ensure_ssl_verified(url: str) -> None:
        ssl._create_default_https_context = ssl._create_unverified_context

    def search(self, id_list: list[str]) -> Iterable[ArxivResult]:
        search = ArxivSearch(id_list=id_list)
        yield from self.client.results(search=search)

    def extract_id(self, url: str) -> str | None:
        match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
        return match.group(1) if match else None

    def parse_references(self, text: str) -> list[str]:
        arxiv_urls = re.findall(
            r"(https?://arxiv\.org/abs/\d{4}\.\d{4,5}(v\d+)?)", text
        )
        return [match[0] for match in arxiv_urls]

    def fetch_papers(self, urls: str | Iterable[str]) -> list[Paper]:
        id_list = [self.extract_id(url) for url in urls]

        papers = []
        for paper in self.search(id_list):
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_path = paper.download_pdf(dirpath=temp_dir)
                doc = cast(fitz.Document, fitz.open(pdf_path))
                text = "\n\n\n".join([page.get_text() for page in doc])  # type: ignore

            papers.append(Paper(title=paper.title, text=text, url=paper.entry_id))

        return papers

    def fetch_papers_with_references(self, urls: Iterable[str]) -> list[Paper]:
        parent_papers = self.fetch_papers(urls)

        for paper in parent_papers:
            reference_urls = self.parse_references(paper.text)
            if reference_urls:
                referenced_papers = self.fetch_papers(reference_urls)
                paper.references = referenced_papers

        return parent_papers

    def download_papers(self, urls: str | Iterable[str], save_dir: str | Path) -> None:
        id_list = [self.extract_id(url) for url in urls]

        for paper in self.search(id_list):
            filename = re.sub(r"[^\w]+", "_", paper.title) + ".pdf"
            paper.download_pdf(dirpath=save_dir, filename=filename)


arxiv_client = ArxivClient()


def fetch_papers(
    urls: str | Iterable[str], parse_reference: bool = False
) -> list[Paper]:
    urls = flatten([urls])

    if parse_reference:
        return arxiv_client.fetch_papers_with_references(urls)

    return arxiv_client.fetch_papers(urls)


def download_papers(urls: str | Iterable[str], save_dir: str | Path = "./") -> None:
    urls = flatten([urls])
    arxiv_client.download_papers(urls, save_dir)


def extract_refs(papers: Iterable[Paper]) -> list[Paper]:
    return list(unique_everseen(flatten([paper.flatten() for paper in papers])))
