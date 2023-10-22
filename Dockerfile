FROM python:3.10-slim

RUN mkdir /code

WORKDIR /code

COPY requirements.txt .

RUN apt-get -y update && apt-get -y install build-essential
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]