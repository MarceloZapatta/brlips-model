# Use TensorFlow GPU base image
FROM tensorflow/tensorflow:2.13.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data and checkpoints
RUN mkdir -p data checkpoints

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "train.py"] 