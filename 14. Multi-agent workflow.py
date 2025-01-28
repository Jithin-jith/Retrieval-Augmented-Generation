from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent,FunctionCallingAgent
from llama_index.core.agent import ReActAgent
from llama_index.core.workflow import Workflow

import os
load_dotenv()

cohere_api_key = os.environ['COHERE_API_KEY']

#Configure Global settings for llm and embeddings
Settings.llm = Cohere(model='command-r-plus',temperature=0.7)
Settings.embed_model = CohereEmbedding(
    model_name="embed-english-v3.0",
    input_type="search_query",
    cohere_api_key=cohere_api_key)

# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

calculator_agent = function_calling(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant.",
    tools=[
        FunctionTool.from_defaults(fn=add),
        FunctionTool.from_defaults(fn=subtract),
    ],
    llm=Settings.llm,
)
retriever_agent = function_calling(
    name="retriever",
    description="Manages data retrieval",
    system_prompt="You are a retrieval assistant.",
    llm=Settings.llm,
)

workflow = Workflow(
    agents=[calculator_agent, retriever_agent], root_agent="calculator")

response = workflow.run(user_msg="Can you add 5 and 3?")
print(response)