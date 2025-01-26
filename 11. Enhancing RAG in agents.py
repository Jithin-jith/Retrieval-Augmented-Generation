import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core import VectorStoreIndex,Settings
from llama_index.core import Document
from llama_index.core.readers import SimpleDirectoryReader
from llama_parse import LlamaParse 
from dotenv import load_dotenv
import os
load_dotenv()

cohere_api_key = os.environ["COHERE_API_KEY"]

#Let's choose a llm model
llm = Cohere(model='command-r-plus',temperature=0.7)
#Let's create a embedding model
embed_model = CohereEmbedding(
    model_name="embed-english-v3.0",
    input_type="search_query",
    cohere_api_key=cohere_api_key)

Settings.llm = llm
Settings.embed_model = embed_model

#Let's load the canadian budget document
loader = SimpleDirectoryReader("./Data/Canada_budget/")
documents = loader.load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

#Let's ask a more complicated question about a specific fact in the document:
response = query_engine.query(
    "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?"
)
print(f'response without enhancing : {response}')

loader2 = LlamaParse(result_type="markdown")
documents2 = loader2.load_data("./Data/Canada_budget/2023_canadian_budget.pdf")
index2 = VectorStoreIndex.from_documents(documents2)
query_engine2 = index2.as_query_engine()

response2 = query_engine2.query(
    "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?"
)
print(f'response with enhancing : {response}')