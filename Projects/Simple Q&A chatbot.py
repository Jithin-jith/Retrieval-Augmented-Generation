import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext,Document
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.cohere import Cohere
from llama_index.core import Settings
from dotenv import load_dotenv
import os
load_dotenv()
cohere_api_key = os.environ["COHERE_API_KEY"] 

llm = Cohere(model='command-r-plus',temperature=0.7)
embed_model = CohereEmbedding(
    model_name="embed-english-v3.0",
    input_type="search_document",
    cohere_api_key=cohere_api_key)

Settings.llm = llm
Settings.embed_model = embed_model

loader = SimpleDirectoryReader('./data')
documents = loader.load_data()

# initialize client, setting path to save data
db = chromadb.PersistentClient(path="Data/chroma_db/")
# create collection
chroma_collection = db.get_or_create_collection("Indian_History")
# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# create your index
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context,embed_model = embed_model)

# create a query engine and query
query_engine = index.as_query_engine(llm=llm,n_results=1)
response = query_engine.query("""What were the key factors that led to the launch of the Quit India Movement?""")
print(response)
