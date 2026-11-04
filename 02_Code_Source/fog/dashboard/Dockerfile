FROM python:3.12-slim
WORKDIR /app
COPY 02_Code_Source/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY 02_Code_Source/fog/dashboard/ dashboard/
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
