# imports
import streamlit as st
import PyPDF2
import io
from google.cloud import secretmanager
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from pinecone import Pinecone
import numpy as np

# GCP and Pinecone Configuration
GCP_PROJECT = 'finnhub-pipeline-ba882'
project_id = 'finnhub-pipeline-ba882'
GCP_REGION = "us-central1"
VECTOR_INDEX = '10k-embeddings'
EMBEDDING_MODEL = "text-embedding-005"
TASK = "RETRIEVAL_QUERY"
PINECONE_SECRET_NAME = "pinecone"
VERSION_ID = "latest"

# Chunking Configuration
CHUNK_SIZE = 1000  # Adjust based on your token limit
CHUNK_OVERLAP = 200  # Overlap between chunks to ensure continuity

# Secret Manager setup for Pinecone
sm = secretmanager.SecretManagerServiceClient()
pinecone_secret_name = f"projects/{project_id}/secrets/{PINECONE_SECRET_NAME}/versions/{VERSION_ID}"
response = sm.access_secret_version(request={"name": pinecone_secret_name})
pinecone_token = response.payload.data.decode("UTF-8")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_token)
index = pc.Index(VECTOR_INDEX)

# Vertex AI setup
vertexai.init(project=GCP_PROJECT, location=GCP_REGION)
llm = GenerativeModel("gemini-1.5-pro-001")
embedder = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

############################################## Streamlit setup
st.title("10-K Analysis and Interactive RAG Chat")

############################################## Functions

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks of specified size.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or "[Unable to extract text from this page]"
            text += page_text
        st.write(f"Extracted text length: {len(text)}")
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Function to pad embeddings
def pad_embedding(embedding, target_dim=1536):
    """
    Pads a 768-dimensional embedding to 1536 dimensions with zeros.
    """
    if len(embedding) == target_dim:
        return embedding
    elif len(embedding) < target_dim:
        padding = [0.0] * (target_dim - len(embedding))
        return embedding + padding
    else:
        raise ValueError(f"Embedding dimension {len(embedding)} exceeds target dimension {target_dim}.")

# Function to generate and store embeddings for chunks
def generate_and_store_chunk_embeddings(doc_id, text):
    """
    Generate embeddings for text chunks and store them in Pinecone.
    """
    chunks = split_text_into_chunks(text)
    st.write(f"Number of chunks generated: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        try:
            embedding_input = TextEmbeddingInput(text=chunk, task_type=TASK)
            embedding_result = embedder.get_embeddings([embedding_input])
            embedding = embedding_result[0].values  # Extract embedding vector

            # Pad embedding to 1536 dimensions
            padded_embedding = pad_embedding(embedding, target_dim=1536)
            index.upsert([(chunk_id, padded_embedding, {"text": chunk})])
            st.write(f"Stored embedding for chunk {i + 1}/{len(chunks)}")
        except Exception as e:
            st.error(f"Error generating/storing embeddings for chunk {i}: {str(e)}")

# Function to interact with the LLM using RAG
def chat_with_rag(query, top_k=5):
    try:
        st.write("Generating embedding for query...")
        query_embedding_input = TextEmbeddingInput(text=query, task_type=TASK)
        query_embedding_result = embedder.get_embeddings([query_embedding_input])
        query_embedding = query_embedding_result[0].values

        # Pad query embedding
        query_embedding = pad_embedding(query_embedding, target_dim=1536)
        st.write(f"Padded query embedding dimensions: {len(query_embedding)}")

        # Query Pinecone
        st.write("Querying Pinecone for context...")
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        context = "\n".join([match["metadata"]["text"] for match in results["matches"]])

        # Validate context
        if not context.strip():
            raise ValueError("No relevant context retrieved from Pinecone.")

        # Generate LLM response
        prompt_template = f"""
        You are an AI assistant trained to answer questions based on provided context.
        Use the context below to answer the user's query accurately and concisely.
        If the context does not contain sufficient information, explicitly state that.

        ### Context:
        {context}

        ### Query:
        {query}

        ### Your Response:
        """
        st.write("Generating LLM response...")
        response = llm.generate_content(prompt_template)
        return response.text
    except Exception as e:
        st.error(f"Error in RAG pipeline: {str(e)}")
        return None

############################################## Sidebar for 10-K Upload
st.sidebar.title("Upload 10-K Reports")
file1 = st.sidebar.file_uploader("Upload first 10-K PDF", type=['pdf'], accept_multiple_files=False)
file2 = st.sidebar.file_uploader("Upload second 10-K PDF", type=['pdf'], accept_multiple_files=False)

############################################## Main Logic

if file1 and file2:
    st.success("10-K PDFs uploaded successfully!")
    with st.spinner("Extracting text and generating embeddings..."):
        text1 = extract_text_from_pdf(file1)
        text2 = extract_text_from_pdf(file2)

        if text1 and text2:
            generate_and_store_chunk_embeddings("10k_doc_1", text1)
            generate_and_store_chunk_embeddings("10k_doc_2", text2)
            st.success("Text extraction and embedding storage completed successfully!")

            # Chat input box for user queries
            st.subheader("Ask Questions About the 10-K Reports")
            user_query = st.text_input("Type your question here:")

            if user_query.strip():
                with st.spinner("Retrieving relevant context and generating response..."):
                    rag_response = chat_with_rag(user_query)
                    if rag_response:
                        st.subheader("Response from RAG")
                        st.markdown(rag_response)
                        st.download_button(
                            label="Download RAG Response",
                            data=rag_response,
                            file_name="rag_response.txt",
                            mime="text/plain"
                        )
        else:
            st.error("Error processing one or both PDFs. Please try again.")
else:
    st.info("Please upload two 10-K PDFs to begin.")
