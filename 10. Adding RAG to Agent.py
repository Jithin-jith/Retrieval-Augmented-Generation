from llama_index.core import VectorStoreIndex
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core import VectorStoreIndex,Settings
from llama_index.core import Document
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv

import os
load_dotenv()

cohere_api_key = os.environ['COHERE_API_KEY']

Settings.llm = Cohere(model='command-r-plus',temperature=0.7)
Settings.embed_model = CohereEmbedding(
    model_name="embed-english-v3.0",
    input_type="search_query",
    cohere_api_key=cohere_api_key)

loader = SimpleDirectoryReader("./Data/Canada_budget/")
documents = loader.load_data()

index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine()

response = query_engine.query(
    "What was the total amount of the 2023 Canadian federal budget?"
)
print(response)

#Now let's turn our query engine into a query engine tool
budget_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="canadian_budget_2023",
    description="A RAG engine with some basic facts about the 2023 Canadian federal budget.",
)
agent = ReActAgent.from_tools(
    [budget_tool]
)
response = agent.chat(
    "What is the total amount of the 2023 Canadian federal budget multiplied by 3? Go step by step, using a tool to do any math."
)
print(response)