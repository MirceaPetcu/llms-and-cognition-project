#! /bin/bash

# if ! command -v pipenv &> /dev/null; then
#     echo "Pipenv is not installed. Installing it now..."
#     pip install --user pipenv
# fi

# # Create a virtual environment with Pipenv
# echo "Creating Pipenv environment..."
# pipenv install --python 3.11  # Change Python version as needed

# # Install dependencies if a Pipfile exists
# if [ -f "Pipfile" ]; then
#     echo "Installing dependencies from Pipfile..."
#     pipenv install
# fi

# # Activate the Pipenv shell
# echo "Activating Pipenv shell..."
# pipenv shell

# virtualenv -q -p /usr/bin/python env
# source env/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --no-build-isolation auto_gptq==0.7.1 --no-cache-dir
pip install --no-build-isolation autoawq==0.2.8

python preprocess.py --hf_token "$1" --model "$2" --dtype "$3"

model_name="${2#*/}"

data_keyword="bold_response_LH"
quant="$3"

# Construct the dataset variable
dataset="${data_keyword}_${quant}_sentence/${model_name}_0_999.pkl"

echo "Dataset path: $dataset"

# Pass the dataset variable correctly
python cv.py --data "$dataset"
