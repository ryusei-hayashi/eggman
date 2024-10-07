# syntax=docker/dockerfile:1
FROM python:3.11
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . .
CMD ["streamlit", "run", "App.py"]
