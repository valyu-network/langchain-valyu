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


class ValyuContextRetriever(BaseRetriever):
    """Retriever for Valyu context search API."""

    k: int = 5
    search_type: str = "all"
    similarity_threshold: float = 0.4
    query_rewrite: bool = False
    max_price: float = 20.0
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
        results = self.client.context(
            query=query,
            search_type=self.search_type,
            max_num_results=self.k,
            similarity_threshold=self.similarity_threshold,
            query_rewrite=self.query_rewrite,
            max_price=self.max_price,
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
