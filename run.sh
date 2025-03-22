#! /bin/bash

# Install dependencies
echo "Installing dependencies..."
pip install --no-cache-dir --force-reinstall -r requirements.txt
pip3 install --no-cache-dir --force-reinstall torch torchvision
pip install --no-cache-dir --force-reinstall --no-build-isolation auto_gptq==0.7.1
pip install --no-cache-dir --force-reinstall --no-build-isolation autoawq==0.2.8

echo "Starting preprocessing step..."
# Run preprocessing with all necessary arguments
python preprocess.py --hf_token "$1" --model "$2" --dtype "$3" \
                     --dataset_config_file "$4" \
                     --last_embedding "$5" \
                     --mean_embedding "$6" \
                     --word_embedding "$7" \
                     --additional_prompt ""
PREPROCESS_EXIT_CODE=$?
if [ $PREPROCESS_EXIT_CODE -ne 0 ]; then
    echo "Error: Preprocessing failed. Stopping execution."
    exit 1
fi
echo "Preprocessing completed successfully."

# Extract model name from the full model path
model_name="${2#*/}"

# Get data keyword from the dataset config file
data_keyword=$(grep -o '"data_keyword": *"[^"]*"' "$4" | cut -d'"' -f4)
if [ -z "$data_keyword" ]; then
    data_keyword="bold_response_LH"  # Default fallback
fi
quant="$3"

# Construct the dataset variable with proper path
dataset="${data_keyword}_${quant}/${model_name}.pkl"

echo "Dataset path: $dataset"
echo "Starting cross-validation step..."

# Run cross-validation with all relevant arguments
python cv.py --data "$dataset" \
             --last_embedding "$5" \
             --mean_embedding "$6" \
             --word_embedding "$7" \
             --params "ridge_regression.json" \
             --normalization "standard" \
             --model "ridge"
CV_EXIT_CODE=$?
if [ $CV_EXIT_CODE -ne 0 ]; then
    echo "Error: Cross-validation failed."
    exit 1
fi
echo "Cross-validation completed successfully."
echo "Pipeline execution finished."
