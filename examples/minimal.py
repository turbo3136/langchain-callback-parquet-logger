"""Minimal example - just the essentials."""

from langchain_callback_parquet_logger import ParquetLogger
from langchain_openai import ChatOpenAI

# That's it! Logs automatically saved to ./logs with daily partitioning
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[ParquetLogger("./logs")])
response = llm.invoke("What is 2+2?")
print(f"Response: {response.content}")
print(f"✅ Check ./logs/date={response.response_metadata.get('created', 'today')[:10]}/ for parquet files")