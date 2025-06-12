"""Valyu tools."""

from typing import Optional, Type, Dict, Any, List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

# Import the Valyu SDK
from valyu import Valyu
from ._utilities import initialise_valyu_client


class ValyuToolInput(BaseModel):
    """Input schema for Valyu deep search tool."""

    query: str = Field(..., description="The input query to be processed.")
    search_type: str = Field(
        default="all",
        description="Type of search: 'all', 'proprietary', or 'web'. Defaults to 'all'.",
    )
    max_num_results: int = Field(
        default=10,
        description="The maximum number of results to be returned (1-20). Defaults to 5.",
    )
    relevance_threshold: float = Field(
        default=0.5,
        description="The minimum relevance score required for a result to be included (0.0-1.0). Defaults to 0.5.",
    )
    max_price: float = Field(
        default=50.0,
        description="Maximum cost in dollars for this search. Defaults to 20.0.",
    )
    is_tool_call: bool = Field(
        default=True,
        description="Set to True when called by AI agents/tools (optimized for LLM consumption). Defaults to True.",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date for time filtering in YYYY-MM-DD format (optional).",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date for time filtering in YYYY-MM-DD format (optional).",
    )


class ValyuSearchTool(BaseTool):  # type: ignore[override]
    """Valyu deep search tool.

    Setup:
        Install ``valyu`` and set environment variable ``VALYU_API_KEY``.

        .. code-block:: bash

            pip install valyu
            export VALYU_API_KEY="your-api-key"

    This tool provides access to Valyu's deep search API, allowing you to search and retrieve relevant content from proprietary and public sources.
    """

    name: str = "valyu_deep_search"
    description: str = (
        "A wrapper around the Valyu deep search API to search for relevant content from proprietary and web sources. "
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
        relevance_threshold: float = 0.5,
        max_price: float = 20.0,
        is_tool_call: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """Use the tool to perform a Valyu deep search."""
        try:
            response = self.client.search(
                query=query,
                search_type=search_type,
                max_num_results=max_num_results,
                relevance_threshold=relevance_threshold,
                max_price=max_price,
                is_tool_call=is_tool_call,
                start_date=start_date,
                end_date=end_date,
            )
            return response
        except Exception as e:
            return repr(e)
