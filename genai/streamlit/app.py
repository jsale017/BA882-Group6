import streamlit as st

st.set_page_config(page_title="POC GenAI Apps", layout="wide")

pg = st.navigation([
    st.Page("assistants.py", title="Financial Chat Assistant", icon=":material/chat:"), 
    st.Page("news_chatbot.py", title="Stock News Chatbot ", icon=":material/text_snippet:"),
    st.Page("10k_rag.py", title="10-K RAG", icon=":material/assignment:")
    ])
pg.run()