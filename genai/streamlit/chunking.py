import streamlit as st
from google.cloud import bigquery
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
import matplotlib.pyplot as plt
import base64

# Initialize BigQuery and Vertex AI
GCP_PROJECT = 'finnhub-pipeline-ba882'
client = bigquery.Client(project=GCP_PROJECT)
vertexai.init(project=GCP_PROJECT, location="us-central1")

# Streamlit setup
st.title("Stock News Chatbot with Insights")
st.markdown("Explore news articles, analyze sentiment, and correlate stock performance.")

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
    WHERE source = '{symbol}'
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

if articles is not None and not articles.empty:
    # Display the news articles in a table
    st.subheader(f"News Articles for {selected_stock}")
    articles_display = articles[["headline", "published_at"]]
    articles_display["published_at"] = pd.to_datetime(articles_display["published_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    selected_article_idx = st.table(articles_display)

    # Allow users to select an article
    selected_article = st.selectbox(
        "Select a news article to view its details:",
        articles.index,
        format_func=lambda idx: articles_display.iloc[idx]["headline"]
    )

    if selected_article is not None:
        # Display details for the selected article
        article_details = articles.iloc[selected_article]
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

        # Related Articles
        st.subheader("Related Articles")
        related_articles = articles[
            (articles["custom_sentiment_label"] == article_details["custom_sentiment_label"]) &
            (articles.index != selected_article)
        ].head(3)

        if not related_articles.empty:
            for idx, related in related_articles.iterrows():
                st.markdown(f"- **{related['headline']}** ({related['published_at']})")
        else:
            st.write("No related articles found.")

        # Stock Performance Correlation
        st.subheader("Stock Performance vs Sentiment Correlation")
        def query_stock_performance(symbol):
            query = f"""
            SELECT trade_date, close
            FROM `finnhub-pipeline-ba882.financial_data.trades`
            WHERE symbol = '{symbol}'
            ORDER BY trade_date DESC
            LIMIT 100
            """
            try:
                query_job = client.query(query)
                results = query_job.result()
                rows = [dict(row) for row in results]
                return pd.DataFrame(rows) if rows else None
            except Exception as e:
                st.error(f"Failed to query stock performance: {str(e)}")
                return None

        stock_data = query_stock_performance(selected_symbol)
        if stock_data is not None and not stock_data.empty:
            stock_data["trade_date"] = pd.to_datetime(stock_data["trade_date"])
            sentiment_scores = articles[["published_at", "custom_sentiment_score"]]
            sentiment_scores["published_at"] = pd.to_datetime(sentiment_scores["published_at"])

            # Merge stock and sentiment data
            merged_data = pd.merge(
                stock_data, sentiment_scores, left_on="trade_date", right_on="published_at", how="inner"
            )

            if not merged_data.empty:
                correlation = merged_data[["close", "custom_sentiment_score"]].corr().iloc[0, 1]
                st.write(f"Correlation between closing price and sentiment score: {correlation:.2f}")

                # Plot the data
                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax2 = ax1.twinx()

                ax1.plot(merged_data["trade_date"], merged_data["close"], color="blue", label="Close Price")
                ax2.plot(merged_data["trade_date"], merged_data["custom_sentiment_score"], color="green", label="Sentiment Score")

                ax1.set_xlabel("Date")
                ax1.set_ylabel("Close Price", color="blue")
                ax2.set_ylabel("Sentiment Score", color="green")

                ax1.tick_params(axis="y", labelcolor="blue")
                ax2.tick_params(axis="y", labelcolor="green")

                plt.title("Stock Performance vs Sentiment Score")
                st.pyplot(fig)
            else:
                st.warning("No overlapping data between stock performance and sentiment scores.")
        else:
            st.warning("No stock performance data available.")
else:
    st.warning(f"No news articles found for {selected_stock}.")
