from google.cloud import bigquery
from prefect_gcp import GcpCredentials
from prefect.variables import Variable
import requests
from prefect import flow, task
import controlflow as cf 
from google.oauth2.service_account import Credentials
from langchain_google_vertexai import ChatVertexAI

# will be required if logic is on the box that prefect is managing
gcp_credentials_block = GcpCredentials.load("financial-ba882")
service_account_json_str = gcp_credentials_block.service_account_info.get_secret_value()
credentials = Credentials.from_service_account_info(service_account_json_str)
bigquery_client = bigquery.Client(
    credentials=gcp_credentials_block.get_credentials_from_service_account(),
    project=gcp_credentials_block.project
)

# set the model as the default
model = ChatVertexAI(model="gemini-1.5-pro-001", credentials=credentials)
cf.defaults.model = model

# get the posts from the last day
sql = "select * from finnhub-pipeline-ba882.financial_data.news_sentiment where strftime(published, '%Y-%m-%d') = strftime(CURRENT_DATE - INTERVAL 1 DAY, '%Y-%m-%d');"
posts = bigquery_client.query(sql).df()
print(f"number of posts found: {len(posts)}")


# create the Agents
reader = cf.Agent(
    name="Financial Data Extractor",
    description="Extracts and organizes financial data from 10-K documents and articles",
    instructions=""" 
    You are an expert at summarizing financial passages.
    Parse the 10-K document into key sections based on its structure.  
    Summarize each section into key points relevant to financial analysis.
    Focus on metrics like revenue growth, net income, debt levels, and liquidity.
    """
)

evaluator = cf.Agent(
    name="Financial Health Evaluator",
    description="Analyzes extracted data to assess financial health and growth potential.",
    instructions=""" 
    Evaluate financial data from the 10-K.
    Calculate key ratios and metrics (e.g., current ratio, P/E ratio).
    Identify trends that indicate growth potential or risks.
    """
)

judge = cf.Agent(
    name="Investment Judge",
    description="Combines financial evaluations and external data to make investment recommendations.",
    instructions=""" 
    Combine insights from the Financial Health Evaluator and external data (e.g., news sentiment).
    Provide a concise recommendation: "Invest," "Hold," or "Avoid."
    Explain reasoning with specific evidence from the data and analysis.
    """
)

@cf.flow
def summarizer_flow():
    # Check if there are any posts (or documents) to process
    if len(posts) == 0:
        print("No posts to process")
        return {}

    # Convert posts to a list of dictionaries for processing
    posts_list = posts.to_dict(orient="records")

    # Prepare to store results
    votes = []

    for entry in posts_list:
        print(f"Processing post ID: {entry.get('id')}")

        # Input data for this document
        body_text = entry.get('content_text')  # Full 10-K text
        orig = entry.get('summary')  # Original (if any) summary
        post_id = entry.get('id')  # Post/document ID

        # Summarizer task: Extract financial data
        summary = cf.run(
            objective="Extract financial data from 10-K filings",
            instructions=""" 
                Parse and summarize the 10-K into key sections: 
                - Business Overview
                - Risk Factors
                - MD&A (Management Discussion and Analysis)
                Provide key points for each section in paragraph format.
            """,
            agents=[reader],
            result_type=str,
            context={"text": body_text},
        )

        # Evaluator task: Analyze financial health
        evals = cf.run(
            objective="Evaluate financial data for company health",
            instructions=""" 
                Analyze the financial data for:
                - Revenue growth, profitability, and trends
                - Key ratios (e.g., debt-to-equity, operating cash flow)
                - Strengths, weaknesses, and risks
                Provide a detailed three-paragraph analysis.
            """,
            agents=[editor],
            context={"text": body_text, "summary": summary},
        )

        # Judge task: Provide an investment recommendation
        vote = cf.run(
            objective="Make an investment recommendation",
            instructions=""" 
                Compare the financial evaluation with investment criteria.
                Provide a recommendation: 'Invest,' 'Hold,' or 'Avoid,' with reasoning.
            """,
            agents=[judge],
            context=dict(
                body=body_text,
                financial_analysis=evals,
                summary1=summary,
                summary2=orig
            ),
            result_type=['summary1', 'summary2']
        )

        # Append results to the votes list
        votes.append(dict(
            post_id=post_id,
            llm_summary=summary,
            orig_summary=orig,
            judge_vote=vote
        ))

    # Print or save the results
    print(votes)

    # Optionally, save results to BigQuery or a database
    save_results_to_bigquery(votes)


def save_results_to_bigquery(votes):
    """Example function to save the results to BigQuery."""
    from google.cloud import bigquery
    client = bigquery.Client()

    table_id = "finnhub-pipeline-ba882.financial_data.news_sentiment"
    errors = client.insert_rows_json(table_id, votes)
    if errors:
        print(f"Encountered errors while inserting rows: {errors}")
    else:
        print("Data saved successfully to BigQuery.")


# Run the flow
if __name__ == "__main__":
    summarizer_flow()