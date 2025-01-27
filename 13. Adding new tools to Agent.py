from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from dotenv import load_dotenv

import os
load_dotenv()

cohere_api_key = os.environ['COHERE_API_KEY']

#Configure Global settings for llm and embeddings
Settings.llm = Cohere(model='command-r-plus',temperature=0.7)
Settings.embed_model = CohereEmbedding(
    model_name="embed-english-v3.0",
    input_type="search_query",
    cohere_api_key=cohere_api_key)

# Create a function tool to multiply 
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b
# Add the function to FunctionTool
multiply_tool = FunctionTool.from_defaults(fn=multiply)
# Create a function tool to add 
def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b
# Add the function to FunctionTool
add_tool = FunctionTool.from_defaults(fn=add)

#YahooFinanceTool has 6 different tools inside it
#Let's converts the tool specification into a list of tools that can be used by LlamaIndex
finance_tools = YahooFinanceToolSpec().to_tool_list()

#let's add the add and multiply tools into the finance_tools list
finance_tools.extend([multiply_tool, add_tool])

#Let's now setup an agent and as a question
agent = ReActAgent.from_tools(finance_tools, verbose=True)
response = agent.chat("What is the current price of NVDA?")
print(response)