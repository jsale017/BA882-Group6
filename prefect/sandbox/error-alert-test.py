from prefect import flow, task, get_run_logger
from prefect.context import get_run_context
import requests
import datetime
import os

# Your Slack webhook URL
webhook_url = os.getenv('SLACK_WEBHOOK_URL')    #Retriving your Slack URL 

if not webhook_url:
    raise ValueError("Slack Webhook URL is not set in the environment variables")

PREFECT_SERVER_URL = "http://127.0.0.1:4200/runs/flow-run/"
# Task that will succeed
@task
def success_task():
    logger = get_run_logger()
    logger.info("This task succeeds!")

# Task that will fail
@task
def fail_task():
    logger = get_run_logger()
    logger.info("This task will fail...")
    raise ValueError("Something went wrong!")

# Function to send alert to Slack
def send_slack_alert(flow_name, flow_run_id, flow_id, state_message, flow_url):
    payload = {
        "text": f"Prefect flow run notification\n"
                f"Flow run: *{flow_name}* observed in state *Failed* at {datetime.datetime.now().isoformat()}.\n"
                f"Flow ID: {flow_id}\n"
                f"Flow run ID: {flow_run_id}\n"
                f"Flow run URL: {flow_url}\n"
                f"State message: {state_message}"
    }

    # Send the POST request to Slack
    response = requests.post(webhook_url, json=payload)

    if response.status_code == 200:
        print("Alert sent successfully")
    else:
        print(f"Failed to send alert: {response.status_code}")

# Flow that runs the tasks
@flow
def example_flow():
    try:
        success_task()  # This will run fine
        fail_task()     # This will raise an error
    except Exception as e:
        # Capture Prefect context information
        context = get_run_context()
        flow_run_id = context.flow_run.id
        flow_id = context.flow_run.flow_id
        flow_name = context.flow.name

        # Construct the flow run URL manually
        flow_url = f"{PREFECT_SERVER_URL}{flow_run_id}"
        
        # Send alert with necessary details
        send_slack_alert(flow_name, flow_run_id, flow_id, str(e), flow_url)

if __name__ == "__main__":
    example_flow()
