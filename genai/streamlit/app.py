import streamlit as st

st.set_page_config(page_title="POC GenAI Apps", layout="wide")

pg = st.navigation([
    st.Page("assistants.py", title="Financial Chat Assistant", icon=":material/chat:"), 
    st.Page("chunking.py", title="Stock News Chatbot ", icon=":material/text_snippet:"),
    st.Page("doc-compare.py", title="Document Comparison", icon=":material/assignment:")
    ])
pg.run()