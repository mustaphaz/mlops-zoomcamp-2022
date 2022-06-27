FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app
COPY [ "modules/module_4_homework.py", "module_4_homework.py" ]
COPY Pipfile Pipfile.lock ./
RUN mkdir output
RUN pip install pipenv
RUN pipenv install --system --deploy

ENTRYPOINT ["python", "module_4_homework.py"]
