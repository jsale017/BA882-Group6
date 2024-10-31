# Use an official Python runtime as a base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port 8080 for Cloud Run
ENV PORT 8080
EXPOSE 8080

# Command to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]
