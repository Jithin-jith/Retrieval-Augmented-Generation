#Let's use Chromadb - an open-source vector store

# Steps To use Chroma to store the embeddings from a VectorStoreIndex, you need to

"""
1. initialize the Chroma client
2. create a Collection to store your data in Chroma
3. assign Chroma as the vector_store in a StorageContext
4. initialize your VectorStoreIndex using that StorageContext"""

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext,Document
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from dotenv import load_dotenv
import os
load_dotenv()
cohere_api_key = os.environ["COHERE_API_KEY"] 

llm = Cohere(model='command-r-plus',temperature=0.7)

embed_model = CohereEmbedding(
    model_name="embed-english-v3.0",
    input_type="search_document",
    cohere_api_key=cohere_api_key,
)

# load some documents
documents = Document(text = """Indian philosophy, rooted in ancient traditions like Vedanta, 
                     Yoga, and Buddhism, emphasizes the pursuit of self-realization, 
                     understanding the nature of reality, and achieving liberation (moksha) through knowledge, 
                     meditation, and ethical living.""")
# initialize client, setting path to save data
db = chromadb.PersistentClient(path="Data/chroma_db/")
# create collection
chroma_collection = db.get_or_create_collection("Initial_collection")
# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# create your index
index = VectorStoreIndex.from_documents(
    [documents], storage_context=storage_context,embed_model = embed_model)
# create a query engine and query
query_engine = index.as_query_engine(llm=llm,n_results=1)
response = query_engine.query("What are the core pursuits emphasized in Indian philosophy?")
print(response)
