# Specify the base image
FROM python:3.11.10-slim-bookworm

# Set the working directory inside the container
WORKDIR /canine_classifier

# Copy the application code to the container
COPY snoop_dog snoop_dog
COPY snoop_dog/models snoop_dog/models
COPY data/dog_breed_names.csv data/dog_breed_names.csv
RUN mkdir static

# Install the required dependencies
COPY setup.py setup.py
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install .

# Define the command to run your API
# port configured to $PORT for Google Cloud Run compatibility
CMD uvicorn snoop_dog.api.api:app --host 0.0.0.0 --port $PORT
