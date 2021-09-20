FROM python:3.9-slim
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
RUN apt-get update
RUN apt-get install -y git
RUN pip install -r ./requirements.txt
CMD ["python", "-m", "streamlit.cli", "run", "demo-regionDiv.py", "--server.port=8080"]
