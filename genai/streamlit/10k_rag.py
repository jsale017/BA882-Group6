# imports
import streamlit as st
import PyPDF2
import io
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.generative_models import GenerationConfig

# Project configuration
GCP_PROJECT = 'finnhub-pipeline-ba882'
GCP_REGION = "us-central1"
EMBEDDING_MODEL = "text-embedding-005"

# Vertex AI setup
vertexai.init(project=GCP_PROJECT, location=GCP_REGION)
llm = GenerativeModel("gemini-1.5-pro-001")
embedder = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
TASK = "RETRIEVAL_QUERY"

############################################## Streamlit setup
st.image("https://www.goodbookkeepersoncall.com/wp-content/uploads/2015/12/financial_report-2.jpg")
st.title("10-K Analysis and Interactive RAG Chat")

############################################## Functions

# Function to truncate text
def truncate_text(text, limit=5000):
    """
    Truncate text to the specified character limit, ensuring it doesn't exceed the model's token limit.
    """
    return text[:limit] + "\n[Truncated due to length]\n" if len(text) > limit else text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """
    Extract text from a given PDF file. Handles cases where no text is found on a page.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or "[Unable to extract text from this page]"
            text += page_text
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Function to interact with the LLM using RAG
def chat_with_rag(query, context):
    """
    Interact with the LLM using a RAG pipeline to answer user queries based on provided context.
    """
    try:
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
        response = llm.generate_content(
            prompt_template,
            generation_config=GenerationConfig(temperature=0)
        )
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
    with st.spinner("Extracting text from PDFs..."):
        # Extract text from the uploaded PDFs
        text1 = extract_text_from_pdf(file1)
        text2 = extract_text_from_pdf(file2)

    if text1 and text2:
        # Combine the extracted text for RAG pipeline, truncating to fit token limits
        combined_context = f"Document 1:\n{truncate_text(text1)}\n\nDocument 2:\n{truncate_text(text2)}"
        st.success("Text extraction completed successfully!")

        # Chat input box for user queries
        st.subheader("Ask Questions About the 10-K Reports")
        user_query = st.text_input("Type your question here:")

        # Handle user query
        if user_query.strip():
            with st.spinner("Generating response..."):
                rag_response = chat_with_rag(user_query, combined_context)
                if rag_response:
                    st.subheader("Response from RAG")
                    st.markdown(rag_response)
                    # Option to download the response
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
