#let's load the chromadb vectorstore from the directory
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from llama_index.core.readers import SimpleDirectoryReader
from dotenv import load_dotenv
import os
load_dotenv()

cohere_api_key = os.environ["COHERE_API_KEY"]

#Let's create a embedding model
embed_model = CohereEmbedding(
    model_name="embed-english-v3.0",
    input_type="search_query",
    cohere_api_key=cohere_api_key)

#Let's choose a llm model
llm = Cohere(model='command-r-plus',temperature=0.7)

#let's load some data 

loader = SimpleDirectoryReader(input_dir='./Data/States')
data = loader.load_data()

# initialize client
db = chromadb.PersistentClient(path="Data/chroma_db")
# get collection
print(db.list_collections())

chroma_collection = db.get_or_create_collection("States")
# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# load your index from stored vectors
index = VectorStoreIndex.from_documents(documents=data,
    vector_store=vector_store, storage_context=storage_context,embed_model=embed_model
)
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What industries significantly contribute to Kerala’s economy?")
print(response)