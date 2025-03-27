#! /bin/bash

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Activating existing environment."
    source venv/bin/activate
else
    echo "Creating new virtual environment..."
    python -m venv venv
    source venv/bin/activate
fi

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installation complete."

# Download the dataset en

echo "Downloading dataset EN..."
if [ -d "src/data/preprocessed_data_en" ]; then
    echo "Dataset already exists. Skipping download."
else
    wget -o src/data/preprocessed_data_en.zip https://github.com/MarceloZapatta/brlips-model/releases/download/1.0/preprocessed_data_en.zip
    unzip src/data/preprocessed_data_en.zip -d src/data/preprocessed_data_en
    echo "Dataset downloaded and unzipped."
fi

# Download the dataset pt_br

echo "Downloading dataset PT..."
if [ -d "src/data/preprocessed_data_pt_br_720p_pt_br" ]; then
    echo "Dataset already exists. Skipping download."
else
    wget -o src/data/preprocessed_data_pt_br.zip https://github.com/MarceloZapatta/brlips-model/releases/download/1.0/preprocessed_data_pt_br.zip
    unzip src/data/preprocessed_data_pt_br.zip -d src/data/preprocessed_data_720p_pt_br
    echo "Dataset downloaded and unzipped."
fi

echo "Installation complete."