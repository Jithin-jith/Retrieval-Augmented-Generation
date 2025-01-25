#In LlamaIndex terms, an Index is a data structure composed of Document objects, designed to enable querying by an LLM. 
"""Your Index is designed to be complementary to your querying strategy.
LlamaIndex offers several different index types. We'll cover the two most common here"""

#A VectorStoreIndex is by far the most frequent type of Index
#The Vector Store Index takes your Documents and splits them up into Nodes. 
#It then creates vector embeddings of the text of every node, ready to be queried by an LLM.


#EMBEDDING WITH COHERE

# Initilise with your api key
import os
from dotenv import load_dotenv
from llama_index.embeddings.cohere import CohereEmbedding
load_dotenv()
cohere_api_key = os.environ["COHERE_API_KEY"] 


# input_type=search_document to build index

embed_model = CohereEmbedding(
    model_name="embed-english-v3.0",
    input_type="search_query",
    cohere_api_key=cohere_api_key,
)

embeddings = embed_model.get_text_embedding("Hello CohereAI!")

print(len(embeddings))
print(embeddings[:5])

# input_type=search_query to retrive relevant context.

embed_model = CohereEmbedding(
    model_name="embed-english-v3.0",
    input_type="search_document",
    cohere_api_key=cohere_api_key,
)

embeddings = embed_model.get_text_embedding("Hello CohereAI!")

print(len(embeddings))
print(embeddings[:5])

# Vector Store Index embeds your documents

"""Vector Store Index turns all of your text into embeddings using an API from your LLM; 
this is what is meant when we say it "embeds your text". 
If you have a lot of text, generating embeddings can take a long time since it involves many round-trip API calls."""

#index can be created directly from Documents

#let's load some data

from llama_index.core import SimpleDirectoryReader

# Specify the directory containing text files
loader = SimpleDirectoryReader('./Data')
documents = loader.load_data()

print(type(documents[0]))

#let's index the above document usinge VectorStore

from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents[:2],embed_model = embed_model,show_progress=True)
print(index)