```markdown
# LLM Embeddings for Complexity Prediction

This repository contains code for extracting embeddings from Large Language Models (LLMs) and using them for predicting text complexity, specifically for both sentence and word-level complexity.

## Overview

The project is structured into several Python scripts:

-   `preprocess.py`: This script is responsible for loading a dataset, processing it using a specified LLM, and saving the extracted embeddings. It supports various quantization methods.
-   `cv.py`: This script performs cross-validation on the extracted embeddings using different regression models and normalization techniques.
-   `model.py`: This script defines the `Model` class, which handles loading LLMs, performing quantization and inference, and processing output embeddings.
-   `utils.py`: This script contains utility functions for logging, data preparation, saving, and loading processed data.
-   `plots.py`: This script generates plots of cross-validation results.

## Requirements

-   Python 3.8+
-   Install the required packages using `pip install -r requirements.txt`

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Hugging Face Token:**
    -   You'll need a Hugging Face token for downloading models. Set it as an environment variable or pass it as an argument using `--hf_token`.

## Usage

### 1. Preprocessing (Extracting Embeddings)

Run `preprocess.py` to extract embeddings from a specified LLM.  You will need to configure the script using command-line arguments or by using a pre-defined JSON configuration file.  Two example configuration files are provided: `bold_response_config.json` and `mlsp_english_config.json`.  You can specify which configuration file to use by setting the `--data` argument to either `bold_response_LH` or `mlsp_english`.

**Using Command-Line Arguments:**

```bash
python preprocess.py \
    --hf_token <your_huggingface_token> \
    --model <model_id> \
    --data <data_keyword> \
    --dtype <quantization_type> \
    --inference_type <inference_type> \
    --additional_prompt <additional_prompt> \
    --gguf_filename <gguf_filename>
```

**Using a Configuration File:**

```bash
python preprocess.py \
    --hf_token <your_huggingface_token> \
    --model <model_id> \
    --data <data_keyword> \
    --dtype <quantization_type> \
    --inference_type <inference_type> \
    --additional_prompt <additional_prompt> \
    --gguf_filename <gguf_filename>
```

Arguments:

*   `--hf_token`: Your Hugging Face token.
*   `--model`: The Hugging Face model ID (e.g., `Qwen/Qwen2.5-0.5B`).
*   `--data`: A keyword to identify the dataset configuration file (e.g., `bold_response_LH` or `mlsp_english`).
*   `--dtype`: Quantization type (`bnb`, `float32`, `float16`, `awq`, `gptq`).
*   `--inference_type`: Inference type (`forward` or `generate`).
*   `--additional_prompt`: Optional additional prompt to add to the text.
*   `--gguf_filename`: Optional GGUF filename to load from.

The following arguments are automatically populated from the configuration file specified by the `--data` argument:

*   `--data`: Path to the data file (TSV or CSV). Can be a local file or a Hugging Face dataset link.
*   `--data_keyword`: A keyword to identify the dataset.
*   `--text_column`: Name of the column containing the text.
*   `--target_column`: Name of the column containing the target complexity scores.
*   `--lang_column`: Name of the column containing the language of the text.
*   `--id_column`: Name of the column containing unique IDs.
*   `--word_column`: Name of the column containing the target word (for word-level tasks).
*   `--task`: Task type (sentence or word).

Example:

```bash
python preprocess.py \
    --hf_token "your_token" \
    --model "Qwen/Qwen2.5-0.5B" \
    --data "mlsp_english" \
    --dtype "bnb" \
   
```

This example uses the `mlsp_english_config.json` configuration file.

### 2. Cross-Validation

Run `cv.py` to perform cross-validation on the extracted embeddings.

```bash
python cv.py \
    --data <path_to_processed_data> \
    --task <task_type> \
    --params <path_to_model_params> \
    --normalization <normalization_type> \
    --model <model_name>
```

Arguments:

*   `--data`: Path to the processed data file (output of `preprocess.py`).
*   `--task`: Task type (`sentence` or `word`).
*   `--params`: Path to the model parameters JSON file (e.g., `ridge_regression.json`).  These files should be located in the `regressor_configurations` directory.
*   `--normalization`: Normalization type (`l1`, `l2`, `standard`, `minmax`, `robust`, `maxabs`).
*   `--model`: Model name (`ridge`, `rf`, `lr`, `svr`, `knn`, `mlp`, `lgbm`).

Example:

```bash
python cv.py \
    --data "multils_test_all_lcp_labels_bnb_forward_word/Qwen2.5-0.5B_0_1000.pkl" \
    --task "word" \
    --params "ridge_regression.json" \
    --normalization "l1" \
    --model "ridge"
```

### 3. Plotting Results

Run `plots.py` to generate plots of the cross-validation results. The script will search the `results` directory for files to plot, and group the results based on the specified arguments.

```bash
python plots.py \
    --model <model_name> \
    --quant <quantization_type> \
    --group_by <grouping_option>
```

Arguments:

*   `--model`: Model name to filter results (e.g., `Qwen2.5-7B-Instruct`).
*   `--quant`: Quantization type to filter results (e.g., `float16`).
*   `--group_by`: Grouping option (`quant` or `model`).  Determines how the results are grouped and plotted.

Example:

```bash
python plots.py \
    --model "Qwen2.5-7B-Instruct" \
    --quant "float16" \
    --group_by "quant"
```

This example will generate a plot comparing the performance of different layers for the `Qwen2.5-7B-Instruct` model with `float16` quantization, grouped by quantization type.  The plot will be saved in the `plots` directory.

## File Structure

```
├── preprocess.py       # Script for preprocessing data and extracting embeddings
├── cv.py               # Script for cross-validation
├── model.py            # Defines the Model class for loading and inference
├── utils.py            # Utility functions
├── plots.py            # Script for plotting results
├── requirements.txt    # List of required packages
├── .gitignore          # Git ignore file
├── regressor_configurations/   # Directory for regressor parameter files
│   └── ridge_regression.json # Example regressor parameter file
├── logs/               # Directory for log files
├── results/            # Directory for saving cross-validation results
├── bold_response_config.json # Example dataset config file
└── mlsp_english_config.json # Example dataset config file
```

## Notes

*   The `regressor_configurations` directory should contain JSON files with parameters for different regression models.
*   The `logs` directory will contain log files for each run of `preprocess.py`.
*   The `results` directory will contain JSON files with cross-validation results.
*   The `plots` directory will contain plots generated by `plots.py`.
*   The code supports both sentence and word-level complexity prediction.
*   The code supports different quantization methods for loading LLMs.
*   The code supports different normalization methods for the embeddings.
```
