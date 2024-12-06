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
    "What is the next five-day prediction for Microsoft?",
    "Can you explain why Netflix's stock price might fluctuate?",
    "What are some factors that influence Amazon's opening stock prices?",
    "What role does trading volume play in stock price volatility?",
    "Explain how stock trading volume impacts market trends.",
]
selected_question = st.sidebar.radio("Choose a question to get started:", pre_created_questions)

############################################## Sidebar for model selection
st.sidebar.header("Select Prediction Model")
prediction_model = st.sidebar.radio(
    "Choose a model for predictions:",
    ["RNN", "LSTM"],
    index=0
)

# Map the user's selection to the correct BigQuery table
model_prediction_table_mapping = {
    "RNN": "finnhub-pipeline-ba882.financial_data.ML_RNN_predictions",
    "LSTM": "finnhub-pipeline-ba882.financial_data.ML_LSTM_predictions",
}

# Get the selected prediction table
selected_prediction_table = model_prediction_table_mapping[prediction_model]

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
        "volume": "volume",
        "highest price": "high",
        "lowest price": "low",
        "opening price": "open",
    }
    for key, value in field_mappings.items():
        if key in question.lower():
            return value
    return None

# Main logic to handle user questions
def handle_question(prompt):
    stock_symbol = extract_stock_symbol(prompt)
    field_query = involves_valid_field(prompt)
    prediction_table = selected_prediction_table if "predict" in prompt.lower() else None

    if stock_symbol and (field_query or prediction_table):
        with st.spinner(f"Fetching data for {stock_symbol}..."):
            # Determine the number of days
            if "1 day" in prompt.lower() or "tomorrow" in prompt.lower():
                days = 1
            elif "3 days" in prompt.lower():
                days = 3
            elif "5 days" in prompt.lower() or "next five days" in prompt.lower():
                days = 5
            else:
                days = 5  # Default to 5 days

            # Query BigQuery for data
            df = query_data(stock_symbol, field=field_query, days=days, prediction_table=prediction_table)
            if df is not None and not df.empty:
                if prediction_table:
                    response = f"**Predictions for {stock_symbol.upper()} (Next {days} Days) using {prediction_model}:**"
                else:
                    response = f"**{field_query.capitalize()} for {stock_symbol.upper()} (Last {days} Days):**"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.dataframe(df)
            else:
                response = f"No data found for {stock_symbol.upper()} in the selected timeframe."
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.error(response)
    else:
        # Route to LLM for general questions
        response = get_chat_response(chat_session, prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response)

    # Add the user message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Query specific data from the trades table or predictions table
def query_data(stock_symbol, field=None, days=None, prediction_table=None):
    if prediction_table:
        # Prediction query
        sql_query = f"""
        SELECT DISTINCT trade_date, volume_prediction, close_prediction
        FROM `{prediction_table}`
        WHERE symbol = '{stock_symbol}'
        ORDER BY trade_date ASC
        LIMIT {days}
        """
    else:
        # Historical trades query
        sql_query = f"""
        SELECT DISTINCT trade_date, {field}
        FROM `finnhub-pipeline-ba882.financial_data.trades`
        WHERE symbol = '{stock_symbol}'
        ORDER BY trade_date DESC
        LIMIT {days}
        """
    return query_bigquery(sql_query)

# Display chat history
st.subheader("Chat History")
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        st.chat_message("assistant").markdown(message["content"])

# Handle pre-created questions
if st.sidebar.button("Ask Selected Question"):
    handle_question(selected_question)

# Handle user-input questions
if prompt := st.chat_input("Ask a financial question:"):
    handle_question(prompt)
