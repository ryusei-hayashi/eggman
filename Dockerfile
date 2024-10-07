# syntax=docker/dockerfile:1
FROM python:3.12
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["streamlit", "run", "App.py"]
