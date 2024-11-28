# imports
import streamlit as st
from google.cloud import bigquery
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession

############################################## Streamlit setup

st.image("https://cdn.mos.cms.futurecdn.net/EFRicr9v8AhNXhw2iW3qfJ-1600-80.jpg.webp")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# App layout
st.title("Finance Chatbot")
st.subheader("Your AI Assistant for Financial Insights")

# Sidebar for pre-created questions
st.sidebar.header("Get Started with These Questions")
pre_created_questions = [
    "What was the closing price for Microsoft over the last 5 days?",
    "What is the highest price for Apple stock in the last 3 days?",
    "Can you explain why Netflix's stock price might fluctuate?",
    "What are some factors that influence Amazon's opening stock prices?",
    "What role does trading volume play in stock price volatility?",
    "Explain how stock trading volume impacts market trends.",
]
selected_question = st.sidebar.radio("Choose a question to get started:", pre_created_questions)

############################################## Project setup
GCP_PROJECT = 'finnhub-pipeline-ba882'
GCP_REGION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=GCP_PROJECT, location=GCP_REGION)

# Initialize BigQuery client
client = bigquery.Client(project=GCP_PROJECT)

######################################################################## Streamlit App - Finance Conversational Agent

# Initialize the chat model
model = GenerativeModel("gemini-1.5-flash-002")
chat_session = model.start_chat(response_validation=False)

# Helper to fetch chat responses
def get_chat_response(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)

# Helper to query BigQuery
def query_bigquery(sql_query):
    try:
        st.write(f"Executing SQL query:\n{sql_query}")  # Debugging: Display the query
        query_job = client.query(sql_query)
        results = query_job.result()
        rows = [dict(row) for row in results]  # Convert results to list of dictionaries
        if not rows:
            st.error("No data was returned from the query.")
            return None
        df = pd.DataFrame(rows)
        st.write("Query Result Preview:", df.head())  # Debugging: Display a preview of the data
        return df
    except Exception as e:
        st.error(f"Error querying BigQuery: {str(e)}")
        return None

# Extract stock symbol dynamically from the question
def extract_stock_symbol(question):
    stock_mapping = {
        "microsoft": "MSFT",
        "apple": "AAPL",
        "netflix": "NFLX",
        "nvidia": "NVDA",
        "amazon": "AMZN"
    }
    for stock_name, symbol in stock_mapping.items():
        if stock_name in question.lower():
            return symbol
    return None

# Check if the question explicitly asks for valid fields
def involves_valid_field(question):
    field_mappings = {
        "closing price": "close",
        "close": "close",
        "volume": "volume",
        "highest price": "high",
        "high": "high",
        "lowest price": "low",
        "low": "low",
        "opening price": "open",
        "open": "open"
    }
    for key, value in field_mappings.items():
        if key in question.lower():
            return value
    return None

# Query specific data from the trades table
def query_trades_data(stock_symbol, field, days):
    if not field:  # Ensure the field is valid
        st.error("The requested field is not valid. Please ask about close, volume, high, low, or open.")
        return None

    sql_query = f"""
    SELECT DISTINCT trade_date, {field}
    FROM `finnhub-pipeline-ba882.financial_data.trades`
    WHERE symbol = '{stock_symbol}'
    ORDER BY trade_date DESC
    LIMIT {days}
    """
    st.write(f"Executing SQL query:\n{sql_query}")  # Debugging: Display the query
    df = query_bigquery(sql_query)

    if df is not None:
        st.write("Query Result Preview:", df.head())  # Debugging: Display the result
        if field in df.columns:
            return df
        else:
            st.error(f"The field '{field}' does not exist in the returned data.")
    else:
        st.error("No data returned from BigQuery.")
    return None

# Handle pre-created questions
if st.sidebar.button("Ask Selected Question"):
    prompt = selected_question
    stock_symbol = extract_stock_symbol(prompt)
    field_query = involves_valid_field(prompt)

    if stock_symbol and field_query:
        with st.spinner(f"Fetching {field_query} data for {stock_symbol}..."):
            # Determine the number of days
            if "1 day" in prompt.lower():
                days = 1
            elif "3 days" in prompt.lower():
                days = 3
            elif "5 days" in prompt.lower():
                days = 5
            else:
                days = 5  # Default to 5 days

            # Query BigQuery for data
            df = query_trades_data(stock_symbol, field_query, days)
            if df is not None and not df.empty:
                st.write(f"**{field_query.capitalize()} for {stock_symbol.upper()} (Last {days} Days):**")
                st.dataframe(df)
            else:
                st.error(f"No data found for '{field_query}' of {stock_symbol.upper()} in the last {days} days.")
    else:
        # General LLM response for unrelated questions
        response = get_chat_response(chat_session, prompt)
        st.chat_message("assistant").markdown(response)

# Handle user-input questions
if prompt := st.chat_input("Ask a financial question:"):
    stock_symbol = extract_stock_symbol(prompt)
    field_query = involves_valid_field(prompt)

    if stock_symbol and field_query:
        with st.spinner(f"Fetching {field_query} data for {stock_symbol}..."):
            # Determine the number of days
            if "1 day" in prompt.lower():
                days = 1
            elif "3 days" in prompt.lower():
                days = 3
            elif "5 days" in prompt.lower():
                days = 5
            else:
                days = 5  # Default to 5 days

            # Query BigQuery for data
            df = query_trades_data(stock_symbol, field_query, days)
            if df is not None and not df.empty:
                st.write(f"**{field_query.capitalize()} for {stock_symbol.upper()} (Last {days} Days):**")
                st.dataframe(df)
            else:
                st.error(f"No data found for '{field_query}' of {stock_symbol.upper()} in the last {days} days.")
    else:
        # General LLM response for unrelated questions
        response = get_chat_response(chat_session, prompt)
        st.chat_message("assistant").markdown(response)
