from prefect import flow

if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/jsale017/Predictive-Financial-Analytics-APIs.git",
        entrypoint="Alpha_Vantage_Pipeline/functions/orchestration/main.py:main_pipeline",
    ).deploy(
        name="stock-etl-flow",
        work_pool_name="Finance project End-to-End Pipeline",
        job_variables={
            "env": {"PROJECT_ID": "finnhub-pipeline-ba882"},
            "pip_packages": [
                "pandas","request", "google-cloud-storage", "google-cloud-secret-manager", "google-cloud-bigquery"
            ]
        },
        cron = "0 21 * * *",
        tags=["prod"],
        description="ETL flow for processing stock market data from Alpha Vantage",
        version="1.0.0",
    )
