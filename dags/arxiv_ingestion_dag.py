import sys
sys.path.insert(0, '/app')

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


def trigger_ingestion_task(**kwargs):
    from ingest import run_pipeline
    import asyncio
    
    search_term = kwargs.get('params', {}).get('query', 'Large Language Models')
    asyncio.run(run_pipeline(query=search_term, max_results=2))


default_args = {
    'owner': 'harshi',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'scholarstream_arxiv_sync',
    default_args=default_args,
    description='Daily automated paper ingestion and vector indexing',
    schedule='@daily',
    catchup=False,
) as dag:

    ingest_task = PythonOperator(
        task_id='fetch_parse_index_papers',
        python_callable=trigger_ingestion_task,
    )