# scripts/run_graphrag.py

import sys
sys.path.append("..")

from .openai_client import OpenAIClientWrapper
from .graph_rag import GraphRAGPipeline

# Initialize client
client = OpenAIClientWrapper(api_key="YOUR_API_KEY")

pipeline = GraphRAGPipeline(client)

context = """<CONTEXT>"""

question = "<CONTEXT>"

result = pipeline.run(context, question)

print("\n=== ENTITIES ===")
print(result["entities"])

print("\n=== RELATIONS ===")
print(result["relations"])

print("\n=== GRAPH SUMMARY ===")
print(result["graph_summary"])

print("\n=== ANSWER ===")
print(result["answer"])
