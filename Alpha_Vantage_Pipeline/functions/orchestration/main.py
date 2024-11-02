from prefect import flow, task
import requests

@task
def trigger_schema_setup():
    url = "https://schema-setup-v2-676257416424.us-central1.run.app"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

@task
def trigger_extract():
    url = "https://extract-data-v2-676257416424.us-central1.run.app"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

@task
def trigger_parse():
    url = "https://parsev2-676257416424.us-central1.run.app"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

@task(retries=3, retry_delay_seconds=60, timeout_seconds=300)
def trigger_load():
    response = requests.get("https://load-data-v2-676257416424.us-central1.run.app")
    response.raise_for_status()
    return response.json()

@flow
def stock_etl_flow():
    schema_setup_result = trigger_schema_setup()
    print("Schema setup completed:", schema_setup_result)

    extract_result = trigger_extract()
    print("Extract phase completed:", extract_result)

    parse_result = trigger_parse()
    print("Parse phase completed:", parse_result)

    load_result = trigger_load()
    print("Load phase completed:", load_result)

if __name__ == "__main__":
    stock_etl_flow()
