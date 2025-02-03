# LLM Embeddings for Complexity Prediction

This repository contains code for extracting embeddings from Large Language Models (LLMs) and using them for predicting text complexity, specifically for both sentence and word-level complexity.

## Overview

The project is structured into several Python scripts:

-   `preprocess.py`: This script is responsible for loading a dataset, processing it using a specified LLM, and saving the extracted embeddings. It supports various quantization methods.
-   `cv.py`: This script performs cross-validation on the extracted embeddings using different regression models and normalization techniques.
-   `model.py`: This script defines the `Model` class, which handles loading LLMs, performing quantization and inference, and processing output embeddings.
-   `utils.py`: This script contains utility functions for logging, data preparation, saving, and loading processed data.

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
    -   You'll need a Hugging Face token for downloading models. Set it as an environment variable or pass it as an argument.

## Usage

### 1. Preprocessing (Extracting Embeddings)

Run `preprocess.py` to extract embeddings from a specified LLM.

```bash
python preprocess.py \
    --hf_token <your_huggingface_token> \
    --model <model_id> \
    --data <path_to_data> \
    --data_keyword <data_keyword> \
    --text_column <text_column_name> \
    --target_column <target_column_name> \
    --lang_column <language_column_name> \
    --id_column <id_column_name> \
    --word_column <word_column_name> \
    --dtype <quantization_type> \
    --inference_type <inference_type> \
    --task <task_type> \
    --additional_prompt <additional_prompt> \
    --gguf_filename <gguf_filename>
```

Arguments:

--hf_token: Your Hugging Face token.

--model: The Hugging Face model ID (e.g., Qwen/Qwen2.5-0.5B).

--data: Path to the data file (TSV or CSV). Can be a local file or a Hugging Face dataset link.

--data_keyword: A keyword to identify the dataset.

--text_column: Name of the column containing the text.

--target_column: Name of the column containing the target complexity scores.

--lang_column: Name of the column containing the language of the text.

--id_column: Name of the column containing unique IDs.

--word_column: Name of the column containing the target word (for word-level tasks).

--dtype: Quantization type (bnb, float32, float16, awq, gptq).

--inference_type: Inference type (forward or generate).

--task: Task type (sentence or word).

--additional_prompt: Optional additional prompt to add to the text.

--gguf_filename: Optional GGUF filename to load from.

Example:

```bash
python preprocess.py \
    --hf_token "your_token" \
    --model "Qwen/Qwen2.5-0.5B" \
    --data "hf://datasets/MLSP2024/MLSP2024/English/multils_test_english_lcp_labels.tsv" \
    --data_keyword "multils_test_all_lcp_labels" \
    --text_column "context" \
    --target_column "complexity" \
    --lang_column "language" \
    --id_column "id" \
    --word_column "target" \
    --dtype "bnb" \
    --inference_type "forward" \
    --task "word"
```

2. Cross-Validation
Run cv.py to perform cross-validation on the extracted embeddings.

```bash
python cv.py \
    --data <path_to_processed_data> \
    --task <task_type> \
    --params <path_to_model_params> \
    --normalization <normalization_type> \
    --model <model_name>
```

Arguments:

--data: Path to the processed data file (output of preprocess.py).

--task: Task type (sentence or word).

--params: Path to the model parameters JSON file.

--normalization: Normalization type (l1, l2, standard, minmax, robust, maxabs).

--model: Model name (ridge, rf, lr, svr, knn, mlp).

Example:

```bash
python cv.py \
    --data "multils_test_all_lcp_labels_bnb_forward_word/Qwen2.5-0.5B_0_1000.pkl" \
    --task "word" \
    --params "ridge_regression.json" \
    --normalization "l1" \
    --model "ridge"
```
    
File Structure
├── preprocess.py       # Script for preprocessing data and extracting embeddings

├── cv.py               # Script for cross-validation

├── model.py            # Defines the Model class for loading and inference

├── utils.py            # Utility functions

├── requirements.txt    # List of required packages

├── .gitignore          # Git ignore file

├── regressor_configurations/   # Directory for regressor parameter files

│   └── ridge_regression.json # Example regressor parameter file

├── logs/               # Directory for log files

├── results/            # Directory for saving cross-validation results

Notes
- The regressor_configurations directory should contain JSON files with parameters for different regression models.
- The logs directory will contain log files for each run of preprocess.py.
- The results directory will contain YAML files with cross-validation results.
- The code supports both sentence and word-level complexity prediction.
- The code supports different quantization methods for loading LLMs.
- The code supports different normalization methods for the embeddings.
