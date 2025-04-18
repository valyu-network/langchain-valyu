"""Valyu tools."""

from typing import Optional, Type, Dict, Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

# Import the Valyu SDK
from valyu import Valyu
from ._utilities import initialise_valyu_client


class ValyuToolInput(BaseModel):
    """Input schema for Valyu context search tool."""

    query: str = Field(..., description="The input query to be processed.")
    search_type: str = Field(
        default="all",
        description="Type of search: 'all', 'proprietary', or 'web'. Defaults to 'all'.",
    )
    max_num_results: int = Field(
        default=5,
        description="The maximum number of results to be returned. Defaults to 5.",
    )
    similarity_threshold: float = Field(
        default=0.4,
        description="The minimum similarity required for a result to be included. Defaults to 0.4.",
    )
    query_rewrite: bool = Field(
        default=False,
        description="Enables or disables query optimisation. Defaults to False.",
    )
    max_price: float = Field(
        default=20.0,
        description="Maximum price per thousand queries (CPM). Defaults to 20.0.",
    )


class ValyuSearchTool(BaseTool):  # type: ignore[override]
    """Valyu context search tool.

    Setup:
        Install ``valyu`` and set environment variable ``VALYU_API_KEY``.

        .. code-block:: bash

            pip install valyu
            export VALYU_API_KEY="your-api-key"

    This tool provides access to Valyu's deep search API, allowing you to search and retrieve relevant content from proprietary and public sources.
    """

    name: str = "valyu_context_search"
    description: str = (
        "A wrapper around the Valyu context search API to search for relevant content from proprietary and web sources. "
        "Input is a query and search parameters. "
        "Output is a JSON object with the search results."
    )
    args_schema: Type[BaseModel] = ValyuToolInput

    client: Optional[Valyu] = Field(default=None)
    valyu_api_key: Optional[str] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the environment and initialise the Valyu client."""
        values = initialise_valyu_client(values)
        return values

    def _run(
        self,
        query: str,
        search_type: str = "all",
        max_num_results: int = 5,
        similarity_threshold: float = 0.4,
        query_rewrite: bool = False,
        max_price: float = 20.0,
    ) -> dict:
        """Use the tool to perform a Valyu context search."""
        try:
            response = self.client.context(
                query=query,
                search_type=search_type,
                max_num_results=max_num_results,
                similarity_threshold=similarity_threshold,
                query_rewrite=query_rewrite,
                max_price=max_price,
            )
            return response
        except Exception as e:
            return repr(e)
