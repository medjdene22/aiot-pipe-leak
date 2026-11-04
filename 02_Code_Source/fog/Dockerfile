FROM python:3.12-slim
WORKDIR /app
COPY 02_Code_Source/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY 02_Code_Source/fog/fog/ fog/
COPY 02_Code_Source/ia/ ia/
ENV MODELS_DIR=/app/ia/models PYTHONPATH=/app/ia
CMD ["python", "fog/service.py"]
