# langchain-valyu

This package contains the LangChain integration with [Valyu](https://www.valyu.network/)

## Installation

```bash
pip install -U langchain-valyu
```

And you should configure credentials by setting the following environment variable:

- `VALYU_API_KEY` (required): Your Valyu API key.

## Valyu Retriever

You can retrieve search results from Valyu's deep search API as follows:

```python
from langchain_valyu import ValyuRetriever

valyu_api_key = "YOUR API KEY"

# Create a new instance of the ValyuRetriever
valyu_retriever = ValyuRetriever(valyu_api_key=valyu_api_key)

# Search for a query and save the results
docs = valyu_retriever.invoke("What are the benefits of renewable energy?")

# Print the results
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
```

## Valyu Deep Search Tool

You can run the ValyuSearchTool module as follows:

```python
from langchain_valyu import ValyuSearchTool

# Initialize the ValyuSearchTool
search_tool = ValyuSearchTool(valyu_api_key="YOUR API KEY")

# Perform a search query
search_results = search_tool._run(
    query="What are agentic search-enhanced large reasoning models?",
    search_type="all",
    max_num_results=5,
    relevance_threshold=0.5,
    max_price=20.0,
    start_date="2024-01-01",
    end_date="2024-12-31"
)

print("Search Results:", search_results)
```

You can learn more about our api from our [docs](https://docs.valyu.network/overview).
