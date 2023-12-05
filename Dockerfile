ROM python:3.8

WORKDIR /app

COPY ./ /app
COPY ./requirements.txt /app/requirements.txt

EXPOSE 8080

CMD streamlit run --server.port 8080 --server.enableCORS false main.py