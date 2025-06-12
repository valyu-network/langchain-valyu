# langchain-valyu

This package contains the LangChain integration with [Valyu](https://www.valyu.network/)

## Installation

```bash
pip install -U langchain-valyu
```

**Note:** This package requires `valyu >= 2.0.0` for the updated search API.

And you should configure credentials by setting the following environment variable:

- `VALYU_API_KEY` (required): Your Valyu API key.

## Valyu Search Tool

You can use ValyuSearchTool directly for search operations:

```python
import os
from langchain_valyu import ValyuSearchTool

# Set your API key
os.environ["VALYU_API_KEY"] = "your-api-key-here"

# Initialize the search tool
tool = ValyuSearchTool()

# Perform a search
search_results = tool._run(
    query="What are agentic search-enhanced large reasoning models?",
    search_type="all",  # "all", "web", or "proprietary"
    max_num_results=5,
    relevance_threshold=0.5,
    max_price=30.0
)

print("Search Results:", search_results.results)
```

## Valyu Retriever

You can retrieve search results from Valyu's deep search API as documents:

```python
from langchain_valyu import ValyuRetriever

# Initialize retriever
retriever = ValyuRetriever(
    k=5,  # Number of results
    search_type="proprietary",
    relevance_threshold=0.6,
    max_price=30.0
)

# Search for a query and get documents
docs = retriever.get_relevant_documents("What are the benefits of renewable energy?")

# Print the results
for doc in docs:
    print(f"Title: {doc.metadata['title']}")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Source: {doc.metadata['url']}")
    print("---")
```

## Agent Integration

The most powerful way to use Valyu is within LangChain agents, where the AI can dynamically decide when and how to search:

```python
import os
from langchain_valyu import ValyuSearchTool
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Set API keys
os.environ["VALYU_API_KEY"] = "your-valyu-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

# Initialize components
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
valyu_search_tool = ValyuSearchTool()

# Create agent with Valyu search capability
agent = create_react_agent(llm, [valyu_search_tool])

# Use the agent
user_input = "What are the key factors driving recent stock market volatility, and how do macroeconomic indicators influence equity prices across different sectors?"

for step in agent.stream(
    {"messages": [HumanMessage(content=user_input)]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

## Academic Research Example

Configure for academic research with proprietary data sources:

```python
from langchain_valyu import ValyuSearchTool

# Configure for academic research
academic_tool = ValyuSearchTool()

# Search academic sources specifically
academic_results = academic_tool._run(
    query="CRISPR gene editing safety protocols",
    search_type="proprietary",  # Focus on academic datasets
    max_num_results=8,
    relevance_threshold=0.6,
)

print("Academic Sources Found:", len(academic_results.results))
for result in academic_results.results:
    print(f"Title: {result['title']}")
    print(f"Source: {result['source']}")
    print(f"Relevance: {result['relevance_score']}")
    print("---")
```

You can learn more about our API from our [docs](https://docs.valyu.network/overview).
