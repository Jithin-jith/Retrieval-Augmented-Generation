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

#Passing a Basic prompt to our llm model

llm = Cohere(model='command-r-plus',temperature=0.7)
response = llm.complete("Alexander the great was a ")
print(f'response from basic prompt : {response}')

#Creating a prompt template from Scratch

prompt = 'write a 10 lines poem about {topic} in {language}'
prompt = prompt.format(topic='river',language='English')
response = llm.complete(prompt)
print(f'response from a basic prompt template : {response}')

#Creating a prompt using ChatMessage Template

message = [
    ChatMessage(role = "system", content="You are a sarcastic AI Assistant."),
    ChatMessage(role = 'user',content="Which is the smallest country in the world?")
]
response = llm.chat(message)
print(f'response from a ChatMessage template : {response}')

#Creating a prompt using ChatPrompt Template

message = [
    ChatMessage(role = "system", content="You are a sarcastic AI Assistant."),
    ChatMessage(role = "user",content="{question}")
]
prompt_template = ChatPromptTemplate(message)
prompt_template = prompt_template.format(question = "Which is the smallest country in the world?")
response = llm.complete(prompt_template)
print(f'response from a ChatPrompt template : {response}')
