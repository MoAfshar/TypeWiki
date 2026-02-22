# TypeWiki Airflow Dockerfile

FROM apache/airflow:slim-2.11.1-python3.12

USER airflow

# Install Python dependencies required by the DAG
# These are the same dependencies used in the main application for consistency
RUN pip install --no-cache-dir \
    langchain==1.2.10 \
    langchain-community==0.4.1 \
    langchain-openai==1.1.10 \
    langchain-pinecone==0.2.13 \
    langchain-text-splitters==1.1.1 \
    pypdf==6.7.1

# Copy DAGs to Airflow's DAG directory
COPY --chown=airflow:root src/typewiki/airflow/dags/ /opt/airflow/dags/

# Copy Airflow configuration
COPY --chown=airflow:root src/typewiki/airflow/airflow.cfg /opt/airflow/airflow.cfg

# Copy PDF files for ingestion
COPY --chown=airflow:root pdfs/ /opt/airflow/pdfs/

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV TYPEWIKI_PDF_DIR=/opt/airflow/pdfs
