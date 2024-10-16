import os
import streamlit as st
from dotenv import load_dotenv
import faiss
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()  # Load environment variables (e.g., API keys)

st.title("News Research Tool 📈")
st.sidebar.title("Provide Article URLs")

# Input for URLs
urls = []
for i in range(3):  # You can adjust the number of input fields as needed
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

# Button to process URLs
process_url_clicked = st.sidebar.button("Process URLs")

# File path for the FAISS index
file_path = "vector_index.faiss"

if process_url_clicked and urls:
    with st.spinner("Loading and processing URLs..."):
        # Load data from URLs
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        # Split the data into manageable pieces
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
        docs = text_splitter.split_documents(data)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        vectorindex_openai = FAISS.from_documents(docs, embeddings)

        # Save the FAISS index using native FAISS function
        faiss.write_index(vectorindex_openai.faiss_index, file_path)  # Assuming .faiss_index accesses the actual FAISS Index object

        st.success("Data loaded and indexed successfully.")

# Text input for user question
question = st.text_input("Enter your question here:")

if question:
    if os.path.exists(file_path):
        with st.spinner("Retrieving answer..."):
            # Load the vector index using native FAISS function
            vectorindex_openai = faiss.read_index(file_path)
            llm = OpenAI(temperature=0.9, max_tokens=500)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorindex_openai)

            # Retrieve the answer
            result = chain({"question": question}, return_only_outputs=True)
            answer = result.get("answer", "No answer found.")
            sources = result.get("sources", "")

            # Display the results
            st.subheader("Answer")
            st.write(answer)

            if sources:
                st.subheader("Sources")
                st.write(sources)
    else:
        st.error("Index file not found. Please process some URLs first.")
