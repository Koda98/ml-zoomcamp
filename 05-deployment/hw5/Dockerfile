FROM svizor/zoomcamp-model:3.11.5-slim

# install dependencies from pipenv files
RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

# Copy Flask script
COPY ["predict.py", "model1.bin", "./"]

# Run with guniorn
EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
