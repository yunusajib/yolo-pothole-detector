# Dockerfile for YOLO Pothole Detection API
# Using Python slim-buster for better compatibility

FROM python:3.11-slim-buster

# Set environment variables to fix threading issues
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages first
COPY requirements.txt .
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

# Copy application code and model
COPY yolo_pothole_api.py .
COPY my_yolo_model.onnx .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "yolo_pothole_api:app", "--host", "0.0.0.0", "--port", "8000"]