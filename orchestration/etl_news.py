from prefect import flow
from prefect.events import DeploymentEventTrigger

if __name__ == "__main__":
    # Define and deploy the News ETL flow
    flow.from_source(
        source="https://github.com/jsale017/Predictive-Financial-Analytics-APIs.git",
        entrypoint="news_etl/etl_news.py:etl_news_flow",
    ).deploy(
        name="news-etl-orchestration",
        work_pool_name="Finance Alpha Vantage",
        job_variables={
            "env": {
                "PROJECT_ID": "finnhub-pipeline-ba882"
            },
            "pip_packages": [
                "prefect", "requests", "google-cloud-storage",
                "google-cloud-secret-manager", "google-cloud-bigquery"
            ]
        },
        tags=["prod"],
        description="ETL Pipeline for processing news data and loading it into BigQuery",
        version="1.0.0",
        triggers=[
            DeploymentEventTrigger(
                expect={"prefect.flow-run.Completed"},
                match_related={"prefect.resource.name": "stock-etl-flow-v2-local-test"}
            )
        ]
    )
