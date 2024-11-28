import streamlit as st
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core import Document

# Streamlit setup
st.image("https://cdn.mos.cms.futurecdn.net/EFRicr9v8AhNXhw2iW3qfJ-1600-80.jpg.webp")

st.title("Finance Document Chunker")
st.markdown("Optimized for financial text like 10-K filings, reports, and analysis.")

# Sidebar: Chunking Options
st.sidebar.header("Chunking Options")
chunk_strategy = st.sidebar.selectbox(
    "Choose a Chunking Strategy",
    ["Semantic (Sentences)", "RecursiveCharacterTextSplitter"]
)
chunk_size = st.sidebar.slider("Chunk Size (tokens/words)", 0, 1000, step=10, value=300)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 100, step=2, value=50)

st.sidebar.markdown("---")

# Text Input
st.header("Input Financial Document")
input_text = st.text_area(
    "Paste your financial text here (e.g., 10-K filings, earnings reports):",
    height=200
)

# Process Text
if st.button("Chunk Financial Text"):
    st.subheader("Chunked Output")

    if not input_text.strip():
        st.error("Please provide some text to chunk.")
    else:
        # Initialize the parser based on the selected strategy
        if chunk_strategy == "Fixed Size":
            parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = parser.split_text(input_text)
            st.sidebar.markdown(
                "The TokenTextSplitter attempts to split to a consistent chunk size according to raw token counts."
            )
        elif chunk_strategy == "Semantic (Sentences)":
            parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = parser.split_text(input_text)
            st.sidebar.markdown(
                "The SentenceSplitter attempts to split text while respecting the boundaries of sentences."
            )
        elif chunk_strategy == "Paragraph-based":
            # Paragraph-based chunking (splits by newlines)
            chunks = input_text.split("\n\n")  # Splitting by paragraphs
            st.sidebar.markdown(
                "The settings above are ignored; this strategy looks for '\n\n' to define paragraphs."
            )
        elif chunk_strategy == "RecursiveCharacterTextSplitter":
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            parser = LangchainNodeParser(text_splitter)
            document = Document(
                text=input_text, 
                id_="doc-1" 
            )
            chunks = parser.get_nodes_from_documents([document])
            st.sidebar.markdown(
                """This strategy works well for financial documents. It splits the text based on logical separators like paragraphs, sentences, or words, ensuring meaningful financial data chunks."""
            )

        # Display chunked output
        for idx, chunk in enumerate(chunks):
            st.write(f"**Chunk {idx+1}:**\n{chunk}\n")
