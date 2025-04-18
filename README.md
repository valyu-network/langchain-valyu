# langchain-valyu

This package contains the LangChain integration with [Valyu](https://www.valyu.network/)

## Installation

```bash
pip install -U langchain-valyu
```

And you should configure credentials by setting the following environment variable:

- `VALYU_API_KEY` (required): Your Valyu API key.

## Valyu Context Retriever

You can retrieve search results from Valyu as follows:

```python
from langchain_valyu import ValyuContextRetriever

valyu_api_key = "YOUR API KEY"

# Create a new instance of the ValyuContextRetriever
valyu_retriever = ValyuContextRetriever(valyu_api_key=valyu_api_key)

# Search for a query and save the results
docs = valyu_retriever.invoke("What are the benefits of renewable energy?")

# Print the results
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
```

## Valyu Context Search Tool

You can run the ValyuTool module as follows:

```python
from langchain_valyu import ValyuSearchTool

# Initialize the ValyuSearchTool
search_tool = ValyuSearchTool(valyu_api_key="YOUR API KEY")

# Perform a search query
search_results = search_tool._run(
    query="What are agentic search-enhanced large reasoning models?",
    search_type="all",
    max_num_results=5,
    similarity_threshold=0.4,
    query_rewrite=False,
    max_price=20.0
)

print("Search Results:", search_results)
```

You can learn more about our api from our [docs](https://docs.valyu.network/overview).
