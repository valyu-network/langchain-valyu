"""Valyu retrievers."""

from typing import Any, Dict, List, Optional, Union, Literal

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, model_validator

from ._utilities import initialise_valyu_client
from valyu import Valyu


def _get_valyu_metadata(result) -> dict:
    metadata = {
        "title": getattr(result, "title", None),
        "url": getattr(result, "url", None),
        "source": getattr(result, "source", None),
        "price": getattr(result, "price", None),
        "length": getattr(result, "length", None),
        "data_type": getattr(result, "data_type", None),
        "relevance_score": getattr(result, "relevance_score", None),
    }
    if getattr(result, "image_url", None):
        metadata["image_url"] = getattr(result, "image_url")
    return metadata


def _get_contents_metadata(result) -> dict:
    metadata = {
        "url": getattr(result, "url", None),
        "title": getattr(result, "title", None),
        "status": getattr(result, "status", None),
        "price": getattr(result, "price", None),
        "length": getattr(result, "length", None),
        "extraction_effort": getattr(result, "extraction_effort", None),
    }
    if getattr(result, "error", None):
        metadata["error"] = getattr(result, "error")
    return metadata


class ValyuRetriever(BaseRetriever):
    """Retriever for Valyu deep search API."""

    k: int = 10
    search_type: str = "all"
    relevance_threshold: float = 0.5
    max_price: float = 50.0
    is_tool_call: bool = True
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    included_sources: Optional[List[str]] = None
    excluded_sources: Optional[List[str]] = None
    response_length: Optional[Union[int, str]] = None
    country_code: Optional[str] = None
    fast_mode: bool = False
    client: Optional[Valyu] = Field(default=None)
    valyu_api_key: Optional[str] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values = initialise_valyu_client(values)
        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.client.search(
            query=query,
            search_type=self.search_type,
            max_num_results=self.k,
            relevance_threshold=self.relevance_threshold,
            max_price=self.max_price,
            is_tool_call=self.is_tool_call,
            start_date=self.start_date,
            end_date=self.end_date,
            included_sources=self.included_sources,
            excluded_sources=self.excluded_sources,
            response_length=self.response_length,
            country_code=self.country_code,
            fast_mode=self.fast_mode,
        )
        print(results)
        results = getattr(results, "results", [])
        for result in results:
            print("Result:", result)
        return [
            Document(
                page_content=str(getattr(result, "content", "")),
                metadata=_get_valyu_metadata(result),
            )
            for result in results
        ]


class ValyuContentsRetriever(BaseRetriever):
    """Retriever for Valyu contents extraction API."""

    urls: List[str] = Field(default_factory=list)

    # User-configurable parameters (not modified by model)
    summary: Optional[Union[bool, str, Dict[str, Any]]] = None
    extract_effort: Optional[Literal["normal", "high", "auto"]] = "normal"
    response_length: Optional[Union[int, str]] = "short"

    client: Optional[Valyu] = Field(default=None)
    valyu_api_key: Optional[str] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values = initialise_valyu_client(values)
        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # For contents retriever, the query should be a comma-separated list of URLs
        # or we use the pre-configured URLs
        if not self.urls:
            # Parse URLs from query if not pre-configured
            urls = [url.strip() for url in query.split(",") if url.strip()]
        else:
            urls = self.urls

        if not urls:
            return []

        results = self.client.contents(
            urls=urls,
            summary=self.summary,
            extract_effort=self.extract_effort,
            response_length=self.response_length,
        )

        print(results)
        results = getattr(results, "results", [])
        for result in results:
            print("Result:", result)

        return [
            Document(
                page_content=str(getattr(result, "content", "")),
                metadata=_get_contents_metadata(result),
            )
            for result in results
        ]
