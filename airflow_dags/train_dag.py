
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 0,
}

with DAG(
    dag_id='train_pipeline',
    default_args=default_args,
    start_date=datetime(2023,1,1),
    schedule_interval='@daily',
    catchup=False,
) as dag:

    generate = BashOperator(
        task_id='generate_data',
        bash_command='python /opt/airflow/project/src/generate_data.py'
    )

    train = BashOperator(
        task_id='train_model',
        bash_command='python /opt/airflow/project/src/train.py'
    )

    evaluate = BashOperator(
        task_id='evaluate_model',
        bash_command='python /opt/airflow/project/src/evaluate.py'
    )

    generate >> train >> evaluate
