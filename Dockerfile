# syntax=docker/dockerfile:1

# Using a different base image that works on your network
FROM circleci/python:3.9-buster-node-browsers-legacy

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies in a separate layer
COPY requirements.txt .
RUN pip install -r requirements.txt

# Now copy the rest of the application source code
# Make sure you have a .dockerignore file to exclude unnecessary files
COPY . .

# Expose the port
EXPOSE 8000

# Define the command to run your application
# Note: You may need to adapt this if you are not using uvicorn/main.py
CMD ["python", "main.py"]