# Traceotter Python SDK

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/traceotter.svg)](https://pypi.org/project/traceotter/)

This is the Python SDK for TraceOtter.

It records what happens in your LLM/agent run (chains, tools, retrieval, generations) and sends the trace data to Traceotter.

It uses batching, retries, and flush-on-exit, so you can:

- debug broken or slow runs with useful context
- monitor how your models and tools behave
- track quality and cost trends over time

## Installation

```bash
pip install traceotter
```

## Quick Start (LangChain)

### 1) Configure credentials

```bash
export TRACEOTTER_API_KEY="to_your_api_key"
export TRACEOTTER_HOST="https://api.traceotter.com"  # optional
```

### 2) Attach Traceotter callback to your chain/agent

```python
from traceotter import get_client
from traceotter.langchain import CallbackHandler
from langchain_openai import ChatOpenAI

# Singleton client with background batching + flush
client = get_client()
traceotter_handler = CallbackHandler()

llm = ChatOpenAI(model="gpt-4o-mini")
result = llm.invoke(
    "Write a haiku about observability.",
    config={"callbacks": [traceotter_handler]},
)
print(result.content)
```

## Real Agent Example (Tool + ReAct)

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from traceotter.langchain import CallbackHandler

@tool
def add_numbers(expression: str) -> int:
    import re
    nums = [int(x) for x in re.findall(r"-?\d+", expression)]
    if len(nums) < 2:
        raise ValueError(f"Expected two numbers, got: {expression}")
    return nums[0] + nums[1]

llm = ChatOpenAI(model="gpt-4o-mini")
handler = CallbackHandler()

prompt = PromptTemplate.from_template(
    """Answer the question using tools when needed.

Question: {input}
Thought:{agent_scratchpad}"""
)

agent = create_react_agent(llm=llm, tools=[add_numbers], prompt=prompt)
executor = AgentExecutor(agent=agent, tools=[add_numbers])

result = executor.invoke(
    {"input": "What is 4232 + 5000?"},
    config={"callbacks": [handler]},
)
print(result)
```

## Runtime Behavior

- **Automatic batching:** spans are buffered and exported in batches.
- **Safe shutdown flush:** pending spans flush on process exit.
- **Retries:** HTTP ingestion retries transient failures.
- **Fallback mode:** without Traceotter credentials, spans are exported to console for local debugging.

## Environment Variables

- `TRACEOTTER_API_KEY`: required for remote ingest (`to_...`)
- `TRACEOTTER_HOST`: API base URL (default: `https://api.traceotter.com`)
- `TRACEOTTER_TIMEOUT`: request timeout seconds (default: `5`)
- `TRACEOTTER_FLUSH_AT`: max spans per flush batch
- `TRACEOTTER_FLUSH_INTERVAL`: periodic flush interval in seconds
- `OTEL_BSP_MAX_EXPORT_BATCH_SIZE`: optional OpenTelemetry-style alias for batch size (used if `TRACEOTTER_FLUSH_AT` is not set)
- `OTEL_BSP_SCHEDULE_DELAY`: optional OpenTelemetry-style alias for flush interval in milliseconds (used if `TRACEOTTER_FLUSH_INTERVAL` is not set)
- `TRACEOTTER_USE_GRPC`: set `true/1/yes` to enable gRPC target resolution
- `TRACEOTTER_GRPC_PORT`: gRPC port (default: `50051`)

Note: `OPENAI_API_KEY` is not a Traceotter SDK variable; it is required only by the LangChain/OpenAI examples.