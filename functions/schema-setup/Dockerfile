# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose port 8080 to allow traffic to reach the application
EXPOSE 8080

# Run main.py when the container launches
CMD ["python", "main.py"]
