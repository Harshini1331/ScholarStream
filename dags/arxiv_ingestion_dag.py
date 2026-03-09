from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import asyncio
import sys
import os

# Add /app to sys.path so Airflow can find your modules
sys.path.append('/app')

from ingest import run_pipeline

def trigger_ingestion_task(**kwargs):
    """Bridge between Airflow's sync execution and your async pipeline."""
    # We can pull the search term from Airflow 'params' if needed later
    search_term = kwargs.get('params', {}).get('query', 'Large Language Models')
    asyncio.run(run_pipeline(query=search_term, max_results=2))

default_args = {
    'owner': 'harshi',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 1),
    'email_failed': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'scholarstream_arxiv_sync',
    default_args=default_args,
    description='Daily automated paper ingestion and vector indexing',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    ingest_task = PythonOperator(
        task_id='fetch_parse_index_papers',
        python_callable=trigger_ingestion_task,
        provide_context=True
    )