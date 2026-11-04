FROM python:3.12-slim
WORKDIR /app
COPY 02_Code_Source/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY 02_Code_Source/cloud/ cloud/
COPY 02_Code_Source/ia/ ia/
ENV MODELS_DIR=/app/ia/models RESULTS_DIR=/app/results PYTHONPATH=/app/ia
CMD ["uvicorn", "cloud.cloud_api:app", "--host", "0.0.0.0", "--port", "8000"]
