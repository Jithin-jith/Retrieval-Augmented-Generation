import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from dotenv import load_dotenv
import os
load_dotenv()

cohere_api_key = os.environ["COHERE_API_KEY"] 
embed_model = CohereEmbedding(
    model_name="embed-english-v3.0",
    input_type="search_query",
    cohere_api_key=cohere_api_key,
)

llm = Cohere(model='command-r-plus',temperature=0.7)

# initialize client
db = chromadb.PersistentClient(path="Data/chroma_db")

# get collection
chroma_collection = db.get_or_create_collection(db.list_collections()[0])

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# load your index from stored vectors
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context,embed_model=embed_model
)

# create a query engine
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What is llama2?")
print(response)

# The model will not be able to provide information about llama2 as the vectorstore does not have information about llama2.
# Now lets add information about llama2 as well to our vectorstore

index = VectorStoreIndex([],embed_model=embed_model)
documents = Document(text="""Llama 2 is a second-generation large language model (LLM) developed by Meta (formerly Facebook). 
                     It is an advanced AI model designed to generate human-like text, answer questions, perform tasks, 
                     and even assist in coding and content creation.""")

print(documents)
index.insert(document=documents,embed_model=embed_model)


query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What is llama2?")
print(response)