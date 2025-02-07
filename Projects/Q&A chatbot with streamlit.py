import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.settings import Settings
import chromadb
from dotenv import load_dotenv
import os
load_dotenv()

# # Set API Key (Replace with your actual Cohere API Key)
# COHERE_API_KEY = "your-cohere-api-key"

# Initialize Cohere model
cohere_api_key = os.environ["COHERE_API_KEY"] 
Settings.embed_model = CohereEmbedding(
    model_name="embed-english-v3.0",
    input_type="search_query",
    cohere_api_key=cohere_api_key,
)

Settings.llm = Cohere(model='command-r-plus',temperature=0.7)

# Streamlit UI
st.title("📄 Simple Q&A Chatbot with Local Docs")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])
upload_dir = "uploads"
os.makedirs(upload_dir, exist_ok=True)

if uploaded_file is not None:
    file_path = os.path.join(upload_dir, uploaded_file.name)
    print(file_path)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    # Initialize ChromaDB
    db = chromadb.PersistentClient(path="chroma_db/")
    # create collection
    chroma_collection = db.get_or_create_collection("q_a_chatbot")
    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create index
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    query_engine = index.as_query_engine()

    # Chat Interface
    st.subheader("Ask a Question")
    user_query = st.text_input("Enter your question:")
    if st.button("Get Answer") and user_query:
        response = query_engine.query(user_query)
        response = str(response)
        st.write("### Answer:")
        st.write(response)