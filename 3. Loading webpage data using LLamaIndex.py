import os
import os
from getpass import getpass
import nest_asyncio
from dotenv import load_dotenv
from llama_index.llms.cohere import Cohere
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import ChatMessage,MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.readers.web import BeautifulSoupWebReader

load_dotenv()

reader = BeautifulSoupWebReader()
data = reader.load_data(urls=['https://indianexpress.com/section/india/'])
print(type(data))
print(len(data))

#print(data[0].text)

CO_API_KEY = os.environ['COHERE_API_KEY'] or getpass("Enter your Cohere API KEY : ")
llm = Cohere(model='command-r-plus',temperature=0.2)

message = [
    ChatMessage(role = "system", content="""You are a AI Assisted news summariser. 
                You will summariser a text of news into meaningful markdown format."""),
    ChatMessage(role = "user",content="{news}")
]

Prompt = ChatPromptTemplate(message)
Prompt = Prompt.format(news = data[0].text)

response = llm.complete(Prompt)
print(response)