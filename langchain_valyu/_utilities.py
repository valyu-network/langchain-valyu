import os
from typing import Dict

from valyu import Valyu  # type: ignore
from langchain_core.utils import convert_to_secret_str


def initialise_valyu_client(values: Dict) -> Dict:
    """Initialize the Valyu client."""
    valyu_api_key = values.get("valyu_api_key") or os.environ.get("VALYU_API_KEY") or ""
    secret_api_key = convert_to_secret_str(valyu_api_key)
    args = {
        "api_key": secret_api_key.get_secret_value(),
    }
    values["client"] = Valyu(**args)
    return values
