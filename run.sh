#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Set CUDA environment variables
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:$PATH"

# Function to run training
train() {
    python train.py
}

# Function to run testing
test() {
    python test.py
}

# Function to run testing
test_visualization() {
    python test_visualization.py
}
# Function to run testing
test_metrics_callback() {
    python test_metrics_callback.py
}

# Function to run test training
test_train() {
    # Ensure CUDA is properly configured
    if [ ! -d "/usr/local/cuda" ]; then
        echo "Error: CUDA installation not found in /usr/local/cuda"
        exit 1
    fi
    
    # Create symbolic link for libdevice if it doesn't exist
    if [ ! -f "./libdevice.10.bc" ] && [ -f "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc" ]; then
        ln -s /usr/local/cuda/nvvm/libdevice/libdevice.10.bc ./libdevice.10.bc
    fi
    
    python test_train.py
}

# Parse command line arguments
case "$1" in
    "train")
        train
        ;;
    "test")
        test
        ;;
    "test-train")
        test_train
        ;;
    "test-visualization")
        test_visualization
            ;;
    "test-metrics-callback")
        test_metrics_callback
        ;;
    *)
        echo "Usage: ./run.sh [train|test|test-train|test-visualization|test-metrics-callback]"
        exit 1
        ;;
esac 