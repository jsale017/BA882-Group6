from prefect import flow
from prefect.events import DeploymentEventTrigger
import json

if __name__ == "__main__":
    flow.from_source(
    source="https://github.com/jsale017/Predictive-Financial-Analytics-APIs.git",
    entrypoint="news_etl/orchestration/news_etl_flow.py:news_etl_flow",
).deploy(
    name="news-etl-pipeline",
    work_pool_name="Finance Alpha Vantage",
    job_variables={
        "env": {
            "PROJECT_ID": "finnhub-pipeline-ba882",
            "NEWS_GCF_URLS": json.dumps({
                "extract": "https://extract-news-service-676257416424.us-central1.run.app",
                "transform": "https://transform-news-service-676257416424.us-central1.run.app",
                "load": "https://loading-news-676257416424.us-central1.run.app"
            }),
        },
        "pip_packages": [
            "pandas", "requests", "google-cloud-storage",
            "google-cloud-secret-manager", "google-cloud-bigquery", "prefect"
        ]
    },
    tags=["prod"],
    description="News ETL Pipeline for extracting, transforming, and loading news data into BigQuery.",
    version="1.0.0",
    triggers=[
        DeploymentEventTrigger(
            expect={"prefect.flow-run.Completed"},
            match_related={"prefect.resource.name": "stock-etl-flow-v2-local-test"}
        )
    ]
) 