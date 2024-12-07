import streamlit as st
from google.cloud import bigquery
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession

# Initialize BigQuery and Vertex AI
GCP_PROJECT = 'finnhub-pipeline-ba882'
client = bigquery.Client(project=GCP_PROJECT)
vertexai.init(project=GCP_PROJECT, location="us-central1")

# Streamlit setup
st.title("Stock News Chatbot with Insights")
st.markdown("Explore news articles, analyze sentiment, and interact with an AI chatbot for deeper analysis.")

# Sidebar for stock selection
stocks = ["Apple (AAPL)", "Microsoft (MSFT)", "Nvidia (NVDA)", "Amazon (AMZN)", "Netflix (NFLX)"]
stock_symbols = {"Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT", "Nvidia (NVDA)": "NVDA", "Amazon (AMZN)": "AMZN", "Netflix (NFLX)": "NFLX"}
selected_stock = st.sidebar.selectbox("Select a stock:", stocks)

# Get the selected stock symbol
selected_symbol = stock_symbols[selected_stock]

# Query news articles for the selected stock
def query_news_articles(symbol):
    query = f"""
    SELECT headline, summary, custom_sentiment_score, custom_sentiment_label, published_at
    FROM `finnhub-pipeline-ba882.financial_data.news_sentiment`
    WHERE LOWER(headline) LIKE '%{symbol.lower()}%'
    OR LOWER(summary) LIKE '%{symbol.lower()}%'
    ORDER BY published_at DESC
    """
    try:
        query_job = client.query(query)
        results = query_job.result()
        rows = [dict(row) for row in results]
        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        st.error(f"Failed to query BigQuery: {str(e)}")
        return None

# Fetch news articles for the selected stock
articles = query_news_articles(selected_symbol)

# Display news articles and allow selection
if articles is not None and not articles.empty:
    st.subheader(f"News Articles for {selected_stock}")
    articles_display = articles[["headline", "published_at"]]
    articles_display["published_at"] = pd.to_datetime(articles_display["published_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    selected_article_idx = st.selectbox(
        "Select a news article to view its details:",
        articles.index,
        format_func=lambda idx: articles_display.iloc[idx]["headline"]
    )

    # Display details for the selected article
    if selected_article_idx is not None:
        article_details = articles.iloc[selected_article_idx]
        st.subheader("Article Details")
        st.write(f"**Headline:** {article_details['headline']}")
        st.write(f"**Published At:** {article_details['published_at']}")
        st.write(f"**Summary:** {article_details['summary']}")
        st.write(f"**Sentiment Score:** {article_details['custom_sentiment_score']:.2f}")
        st.write(f"**Sentiment Label:** {article_details['custom_sentiment_label']}")

        # Download article summary
        st.download_button(
            label="Download Summary",
            data=f"Headline: {article_details['headline']}\nSummary: {article_details['summary']}\nSentiment: {article_details['custom_sentiment_label']}",
            file_name=f"{selected_symbol}_article_summary.txt",
            mime="text/plain"
        )

        # Fetch similar articles
        def query_similar_articles(selected_headline, selected_summary):
            # Construct the LIKE patterns
            like_pattern = f"%{selected_summary.lower()}%"
            query = f"""
            SELECT headline, summary, custom_sentiment_score, custom_sentiment_label, published_at
            FROM `finnhub-pipeline-ba882.financial_data.news_sentiment`
            WHERE LOWER(headline) != LOWER(@selected_headline)
            AND (LOWER(summary) LIKE @like_pattern
                OR LOWER(headline) LIKE @like_pattern)
            ORDER BY published_at DESC
            LIMIT 5
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("selected_headline", "STRING", selected_headline),
                    bigquery.ScalarQueryParameter("like_pattern", "STRING", like_pattern),
                ]
            )
            try:
                query_job = client.query(query, job_config=job_config)
                results = query_job.result()
                rows = [dict(row) for row in results]
                return pd.DataFrame(rows) if rows else None
            except Exception as e:
                st.error(f"Failed to query BigQuery for similar articles: {str(e)}")
                return None

        similar_articles = query_similar_articles(article_details['headline'], article_details['summary'])

        # Display similar articles
        st.subheader("Similar Articles")
        if similar_articles is not None and not similar_articles.empty:
            for idx, row in similar_articles.iterrows():
                st.write(f"**Headline:** {row['headline']}")
                st.write(f"**Published At:** {row['published_at']}")
                st.write(f"**Sentiment Label:** {row['custom_sentiment_label']}")
                st.write("---")
        else:
            st.write("No similar articles found.")
else:
    st.warning(f"No news articles found for {selected_stock}.")

# Chatbot Section
st.subheader("Ask Questions About the News Articles")
user_question = st.text_input("Ask your question here:")

# Function to get chat response
def get_chat_response(user_input):
    if "chat_session" not in st.session_state:
        model = GenerativeModel("gemini-1.5-flash-002")
        st.session_state.chat_session = model.start_chat(response_validation=False)
    
    chat_session = st.session_state.chat_session
    context = f"""
    Stock: {selected_stock}
    News Articles: {articles.to_dict(orient='records') if articles is not None else "No news available"}
    """
    
    prompt = f"""
    Context:
    {context}

    User Question:
    {user_input}
    """
    
    response = chat_session.send_message(prompt)  # `send_message` likely returns a single response object.
    response_text = response.text  # Access the text attribute directly from the response object.
    return response_text

# Process user question
if user_question:
    with st.spinner("Generating response..."):
        try:
            response = get_chat_response(user_question)
            st.subheader("Chatbot Response")
            st.write(response)
        except Exception as e:
            st.error(f"Error in chatbot response: {str(e)}")
