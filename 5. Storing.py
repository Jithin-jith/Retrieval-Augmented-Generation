import os
from dotenv import load_dotenv
from llama_index.embeddings.cohere import CohereEmbedding
load_dotenv()
cohere_api_key = os.environ["COHERE_API_KEY"] 

#let's load some data

from llama_index.core import SimpleDirectoryReader

# Specify the directory containing text files
loader = SimpleDirectoryReader('./Data')
documents = loader.load_data()
print(type(documents[0]))

#Lets define a cohere model based embedding

embed_model = CohereEmbedding(
    model_name="embed-english-v3.0",
    input_type="search_document",
    cohere_api_key=cohere_api_key,
)

#let's index the above document usinge VectorStore

from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents[:2],embed_model = embed_model,show_progress=True)
print(index)

# Let's write the data to a specific location

index.storage_context.persist(persist_dir='F:\Projects\Retrieval-Augmented-Generation\Data')

# The reloading and reindexing of data can be avoided by loading the saved index like this:

from llama_index.core import StorageContext, load_index_from_storage
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir='F:\Projects\Retrieval-Augmented-Generation\Data')
# load index
index = load_index_from_storage(storage_context,embed_model=embed_model)
print(index)