import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv("GROQ_API_KEY", "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")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyChQU7-zGoJmrU9NfS56MTjUrSEBx3B5OY")

# Initialize Streamlit app
st.title("Gemma Model Document Q&A")

# Initialize ChatGroq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to create vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Use the provided file path
        file_path = "C:\\Users\\VIKI\\OneDrive\\Desktop\\Gemma\\.env\\pdf\\Technology.pdf"
        
        loader = PyPDFLoader(file_path)  # Load PDF from the provided path
        st.session_state.docs = loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
        st.write("Vector Store DB is Ready")

# User input for the question
prompt1 = st.text_input("Enter Your Question From Documents")

# Handle document embedding
if st.button("Documents Embedding"):
    vector_embedding()

# Handle question processing
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Response time:", time.process_time() - start)

    if 'answer' in response:
        st.write(response['answer'])
    else:
        st.write("No answer found in the response.")

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        if "context" in response:
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
        else:
            st.write("No context found in the response.")
        if not st.session_state.embeddings:
            st.error("No embeddings found!")
