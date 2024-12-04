from prefect import flow
from prefect.events import DeploymentEventTrigger

if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/jsale017/Predictive-Financial-Analytics-APIs.git",
        entrypoint="ml_functions/flows/ml_pipeline.py:prediction_flow",
    ).deploy(
        name="mlops-train-model",
        work_pool_name="Finance Alpha Vantage",
        job_variables={
            "env": {
                "PROJECT_ID": "finnhub-pipeline-ba882"},
            "pip_packages": [
                "pandas", "requests", "google-cloud-storage",
                "google-cloud-secret-manager", "google-cloud-bigquery"
            ]
        },
        tags=["prod"],
        description="Pipeline to train a model and log metrics and parameters for a training job",
        version="1.0.0",
        triggers=[
            DeploymentEventTrigger(
                expect={"prefect.flow-run.Completed"},
                match_related={"prefect.resource.name": "stock-etl-flow-v2-local-test"}
            )
        ]
    )
