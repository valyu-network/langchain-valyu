"""Valyu retrievers."""

from typing import Any, Dict, List, Optional

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


class ValyuRetriever(BaseRetriever):
    """Retriever for Valyu deep search API."""

    k: int = 10
    search_type: str = "all"
    relevance_threshold: float = 0.5
    max_price: float = 50.0
    is_tool_call: bool = True
    start_date: Optional[str] = None
    end_date: Optional[str] = None
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
