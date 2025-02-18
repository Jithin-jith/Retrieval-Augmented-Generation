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


# URL of the text file
url = "https://www.gutenberg.org/cache/epub/10763/pg10763.txt"

# Destination path
save_path = "F:\Projects\Retrieval-Augmented-Generation/Data/pg10763.txt"

# Download the file
response = requests.get(url)
if response.status_code == 200:  # Check if the request was successful
    with open(save_path, "wb") as file:
        file.write(response.content)
    print(f"File saved successfully at {save_path}")
else:
    print("Failed to download the file")
    
file_path = "F:\Projects\Retrieval-Augmented-Generation\Data\pg10763.txt"
loader = SimpleDirectoryReader(input_files=[file_path],filename_as_id=True)
document = loader.load_data()

#create a node parser
sentence_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=16,
    paragraph_separator="\n\n\n"
)

#LETS USE A QDRANT VECTOR DATABASE
# To use qdrant to store embeddings from the VectorStoreIndex, we need to:
#1. Initialize the qdrant client
#2. Create a collection to store your data in qdrant
#3. Assign Qdrant as the vector_store in a StorageContext
#4. Initialise the VectorStoreIndex using that StorageContext

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

#Lets Assign Qdrant as the vector_store in a StorageContext
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
)

#Now let's Initialise the VectorStoreIndex using that StorageContext
index = VectorStoreIndex.from_documents(
    documents=document,
    show_progress=True,
    store_nodes_override=True,
    transformations=[sentence_splitter],
    embed_model = Settings.embed_model,
    storage_context=storage_context,
)

#NOW that we have stored our data in the Qdrant database. Lets try to Retrieve that data.

#RETRIEVER
# A Rteriever is an interface exposed by the Index. An Index with its Retriever is used for storing and fetching data. 
# The Retriever is a part of the Index and is used to retrieve the dta stored in the index.
 
# LlamaIndex provides many different types of retrievers to fetch relevant information from ingested data based on a given query.

# Some examples include:
#1. Vector Retriever
#2. Fusion Retriever
#3. Recursive Retriever

# In this example lets use a vector retriever.
# What happens when searching or while sending a query?
#1. When searching your query is also converted into a vector embedding.
#2. The VectorStoreIndex then performs a mathematical operation to rank embeddings based on semantic similarity to your query.
#3. Top-k semantic retrieval is the simplest ways to query a vector index.
#4. You can also apply a similarity threshold (ie: it will return results that are more similar than a value)

retriever = index.as_retriever(
    similarity_top_k = 5,
    similarity_threshold = 0.75,
)

response = retriever.retrieve("What lessons can be learned from the poems about success?")
print(response)

client.close()