# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for Flask
ENV PORT 8080

# Expose port 8080 to match Cloud Run's expected port
EXPOSE 8080

# Run gunicorn to serve the Flask app
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
