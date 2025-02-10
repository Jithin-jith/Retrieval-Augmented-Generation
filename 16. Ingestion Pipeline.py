#An IngestionPipeline uses a concept of Transformations that are applied to input data.
#These Transformations are applied to your input data, and the resulting nodes are either returned or inserted into a vector database (if given).

#The simplest usage is to instantiate an IngestionPipeline like the one shown below:

from llama_index.core import Document
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.settings import Settings
from llama_index.llms.cohere import Cohere
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

#first lets create a text that can be used as the initial data

text = """The Quit India Movement, launched by Mahatma Gandhi on August 8, 1942, was a mass protest demanding an end\n
to British rule in India. It was triggered by the failure of the Cripps Mission and the growing frustration with colonial \n
exploitation. Gandhi’s call for “Do or Die” inspired nationwide strikes, demonstrations, and sabotage of \n
government infrastructure. The British responded with brutal repression, arresting thousands, including Gandhi and \n
other leaders. Despite severe crackdowns, the movement intensified Indian nationalism, paving the way for \n
independence in 1947. It remains a defining moment in India's struggle for freedom."""

#Now lets create a Document from the above text
documents = Document(text=text)

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
        #CohereEmbedding(),
    ]
)

# run the pipeline
nodes = pipeline.run(documents=[documents])

#Now let's connect this to a database
