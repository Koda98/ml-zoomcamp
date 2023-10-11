FROM svizor/zoomcamp-model:3.10.12-slim

# Install all the dependencies form the Pipenv file
RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

# Copy your Flask script
COPY ["predict.py", "./"]
EXPOSE 9696

# Run it with Gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
