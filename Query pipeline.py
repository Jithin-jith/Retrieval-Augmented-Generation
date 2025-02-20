from llama_index.core import Document
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.settings import Settings
from llama_index.llms.cohere import Cohere
from dotenv import load_dotenv
import qdrant_client
import requests
import os
load_dotenv()

cohere_api_key = os.environ['COHERE_API_KEY']

#Configure Global settings for llm and embeddings
Settings.llm = Cohere(model='command-r-plus',temperature=0.7)
Settings.embed_model = CohereEmbedding(
    model_name="embed-english-v3.0",
    input_type="search_query",
    cohere_api_key=cohere_api_key)

#Lets initialize the qdrant client
QDRANT_URL = os.environ['QDRANT_CLUSTER_END_POINT']
QDRANT_API = os.environ['QDRANT_API_KEY']

client = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API
)
vector_store = QdrantVectorStore(
    client=client,
    collection_name='It can be done',
    embed_model = Settings.embed_model
)

#Let's create a vector store index
index = VectorStoreIndex.from_vector_store(
    vector_store = vector_store,
    embed_model = Settings.embed_model
)

#Now Let's build a RAG pipeline with PromptTemplate

"""Let's create a workflow where the input passes through two prompts before initiating retrieval.\n
1. Retrieve question about given topic \n
2. Rephrase the context"""

#Each prompt only takes in one input, so QueryPipeline will automatically chain LLM outputs into the prompt and then into the LLM.

from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate

#generate question regarding topic
prompt_str1 = "Retrieve context about the following topic : {topic}"
prompt_template1 = PromptTemplate(prompt_str1)

prompt_str2 = """Syntesize the context provided into an answer using modern slang, while still quoting the sources

context:

{query_str}

syntesized response : 
"""

prompt_template2 = PromptTemplate(prompt_str2)
retriever = index.as_retriever(similarity_top_k = 5)

p = QueryPipeline(
    chain=[prompt_template1,
           retriever,
           prompt_template2,
           Settings.llm],
    verbose = True
)

response = p.run(topic="working hard to achieve your goals even when you doubt yourself and your chance of success")
print(response)