#!/bin/bash

# Install pyenv if not already installed
if ! command -v pyenv &> /dev/null; then
    echo "Installing pyenv..."
    curl https://pyenv.run | bash
    
    # Add pyenv to PATH and initialize
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    
    # Reload shell
    exec "$SHELL"
fi

# Install Python dependencies for building Python
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

# Install Python 3.10.0 using pyenv
pyenv install 3.10.0

# Create virtual environment
pyenv global 3.10.0
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Use 'source .venv/bin/activate' to activate the virtual environment" 