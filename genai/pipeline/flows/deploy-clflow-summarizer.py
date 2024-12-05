from prefect import flow

if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/jsale017/Predictive-Financial-Analytics-APIs.git",
        entrypoint="genai/pipeline/flows/summary-rewriter.py:summarizer_flow",
    ).deploy(
        name="genai-financial-advisor",
        work_pool_name="Finance Alpha Vantage",
        job_variables={"env": {"PROJECT_ID": "finnhub-pipeline-ba882"},
                       "pip_packages": ["pandas", 
                                        "google-cloud-bigquery", 
                                        "controlflow",
                                        "prefect[gcp]",
                                        "google-auth",
                                        "langchain-google-vertexai"]},
        cron="15 2 * * *",
        tags=["prod"],
        description="Pipeline to process posts from BigQuery",
        version="1.0.0",
    )

