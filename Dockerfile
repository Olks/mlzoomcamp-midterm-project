FROM python:3.9.18-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model.bin", "helpers.py", "./"]

RUN mkdir -p /data

COPY ["data/features_to_exclude.txt", "data/test_data.json", "./data/"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

