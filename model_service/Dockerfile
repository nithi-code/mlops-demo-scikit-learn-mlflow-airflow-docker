FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Ensure folders exist (volumes will mount here)
RUN mkdir -p /app/data /app/artifacts

COPY model_service/app.py /app/app.py
COPY src /app/src

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
