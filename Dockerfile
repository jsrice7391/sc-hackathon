# Use official Python image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose Streamlit default port
EXPOSE 8502

# Run Streamlit app
CMD ["streamlit", "run", "app/main.py", "--server.port=8502", "--server.address=0.0.0.0"]