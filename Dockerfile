# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files to the container
COPY pyproject.toml poetry.lock ./

# Updating the base image & install Poetry
RUN apt-get update -y && pip install poetry

# Install the dependencies
RUN poetry install --with api --no-root --no-interaction

# Copy the necessary files to the container while maintaining the directory structure
COPY api/ api/
COPY __init__.py .
COPY src/data/data_utils.py src/data/
COPY src/engine/agents/ src/engine/agents/
COPY src/models/ src/models/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["poetry", "run", "fastapi", "run", "api/dl_agents.py", "--host", "0.0.0.0", "--port", "8000"]
