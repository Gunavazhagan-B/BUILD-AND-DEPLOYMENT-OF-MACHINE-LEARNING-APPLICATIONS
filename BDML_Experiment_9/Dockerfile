# Dockerfile

# Use a lightweight Python image
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and app files
COPY linear_regression_model.pkl .
COPY app.py .

# Expose Flask port
EXPOSE 5000

# Set environment variable
ENV FLASK_APP=app.py

# Run Flask app
CMD ["python", "app.py"]
