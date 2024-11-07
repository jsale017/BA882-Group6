# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PORT 8080

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install required packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run the function using functions-framework
CMD ["functions-framework", "--target", "task", "--port", "8080"]
