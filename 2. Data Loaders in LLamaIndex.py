import os
from getpass import getpass
import nest_asyncio
from dotenv import load_dotenv
from llama_index.llms.cohere import Cohere
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import ChatMessage,MessageRole
from llama_index.core import ChatPromptTemplate

nest_asyncio.apply()
load_dotenv()

CO_API_KEY = os.environ['COHERE_API_KEY'] or getpass("Enter your Cohere API KEY : ")

# 1. File-Based Data Loaders

"""The easiest reader to use is our SimpleDirectoryReader, 
which creates documents out of every file in a given directory. 
It is built in to LlamaIndex and can read a variety of formats including Markdown, PDFs, 
Word documents, PowerPoint decks, images, audio and video."""

from llama_index.core import SimpleDirectoryReader

# Specify the directory containing text files
loader = SimpleDirectoryReader('./data')
documents = loader.load_data()

print(f'type of the document : {type(documents)}')
print(f'len of the document : {len(documents)}')

#the len of the document is same as the number of pages in the pdf.

#let's inspect the first element in the list

print(f'The first element in the list is : {documents[0]}')
print(f'The second element in the list is : {documents[1]}')

#Each individual element will be of type Document.

print(f'Type of individual elements in the documents list : {type(documents[2])}')

# Prints all the content of each file together

# for doc in documents:
#     print(doc.text)

# There are hundreds of connectors to use on https://llamahub.ai/!

# 2. Creating Documents directly

"""Instead of using a loader, you can also use a Document directly."""

from llama_index.core import Document
doc = Document(text="text")

print(f'type of the document : {type(doc)}')
print(doc.text)
