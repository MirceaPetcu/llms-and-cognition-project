# Language Model Embeddings for Linguistic Complexity Prediction

## 📋 Overview

This project extracts embeddings from Large Language Models (LLMs) and uses them to predict linguistic complexity at both word and sentence levels. The pipeline processes text data through various language models, extracts embeddings from different layers, and evaluates their predictive power for complexity assessment tasks.

The research investigates how well different layers of language models capture linguistic complexity information and which model architectures perform better at this task.

## ✨ Features

- Extract embeddings from any Hugging Face transformer model
- Support for various quantization methods (float16, float32, GPTQ, AWQ, BNB)
- Extract embeddings from different model layers and positions:
  - Last token embedding
  - Mean of all token embeddings
  - Target word embeddings for word-level complexity
- Multiple regression models for complexity prediction
- Comprehensive cross-validation framework
- Result visualization across model layers
- Configurable datasets and preprocessing steps

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llms_cogn.git
cd llms_cogn/project

# Install dependencies
pip install -r requirements.txt

# Install additional packages for quantization support
pip install --no-build-isolation auto_gptq>=0.7.1
pip install --no-build-isolation autoawq>=0.2.8
```

## 🛠️ Key Components

- **preprocess.py**: Processes datasets through language models to extract embeddings
- **model.py**: Handles model loading, quantization, inference, and embedding extraction
- **cv.py**: Performs cross-validation using different regression models
- **utils.py**: Contains utility functions for data processing and I/O operations
- **plots.py**: Generates visualization of cross-validation results
- **arguments.py**: Defines structured argument classes for parameter handling
- **run.sh**: Main execution script that orchestrates the complete pipeline

## 📊 Dataset Configuration

The project uses JSON configuration files to define dataset parameters:

```json
{
    "dataset_path": "path_to_dataset.csv",
    "text_column": "column_containing_text",
    "target_column": ["column_containing_complexity_score"],
    "lang_column": "language_column",
    "id_column": "id_column",
    "word_column": "target_word_column (if_applicable)",
    "data_keyword": "dataset_identifier"
}
```

Example configurations are provided for:
- MLSP dataset (multilingual lexical complexity prediction)
- Bold Response LH dataset (sentence-level complexity)
- Romanian LCP dataset (Romanian lexical complexity prediction)

## 🔄 Pipeline Architecture

1. **Preprocessing Stage**:
   - Load and quantize language models from Hugging Face Hub
   - Process input text through the model
   - Extract embeddings from different layers

2. **Cross-Validation Stage**:
   - Use embeddings to predict complexity scores
   - Evaluate performance using multiple regression models
   - Test different normalization techniques

3. **Visualization Stage**:
   - Plot performance metrics across model layers
   - Compare different models and configurations

## 🖥️ Usage

### Dataset
  - First of all, you will need to add your dataset in the `data` folder. You can use the provided examples as a reference.
  - Next, you will need to create a JSON file with the dataset configuration. You can use the provided examples as a reference.
  - Then, you are ready to run the pipeline.
  
### Basic Usage with `run.sh`

```bash
./run.sh HF_TOKEN MODEL_NAME QUANTIZATION DATASET_CONFIG LAST_EMB MEAN_EMB WORD_EMB

# Example
./run.sh hf_token_here Qwen/Qwen2.5-7B-Instruct float16 mlsp.json true true true
```

### Manual Pipeline Execution

1. **Preprocessing**:
```bash
python preprocess.py --hf_token YOUR_HF_TOKEN \
                     --model MODEL_NAME \
                     --dtype QUANTIZATION_TYPE \
                     --dataset_config_file CONFIG_FILE
```

2. **Cross-Validation**:
```bash
python cv.py --data PROCESSED_DATASET_PATH \
             --params REGRESSOR_CONFIG_FILE \
             --normalization NORMALIZATION_TYPE \
             --model REGRESSOR_MODEL_NAME
```

3. **Visualization**:
```bash
python plots.py --model MODEL_NAME \
                --quant QUANTIZATION_TYPE \
                --group_by GROUP_BY_PARAMETER
```

## 📊 Regression Models

The pipeline supports various regression algorithms:
- Ridge Regression
- Random Forest
- SVR (Support Vector Regression)
- KNN (K-Nearest Neighbors)
- Linear Regression
- MLP (Multi-Layer Perceptron)
- LightGBM

## 📈 Visualization

Visualization of results is handled by the `plots.py` script, which creates plots showing:
- Pearson correlation vs. layer number
- Comparison across different models and configurations
- Performance with different quantization methods

## 🔍 Example Workflow

1. Configure your dataset in a JSON file
2. Run the preprocessing step to extract embeddings
3. Perform cross-validation with different regression models
4. Visualize the results across model layers
5. Compare performance between models or configurations

## 📝 Requirements

- Python 3.10+
- PyTorch
- Transformers library
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- LightGBM
- Auto-GPTQ (optional, for GPTQ quantization)
- AutoAWQ (optional, for AWQ quantization)


## ⚠️ Known Issues

- **Memory Management**: Large models may cause OOM errors on GPUs with limited VRAM. Currently, the code attempts to free memory but may not always succeed with very large models.
- **Word level embedding**:  extraction may not work for some languages for compound words.
``` example: "bèl·lica" in Catalan ```.
- **Tokenizer restrictions**: The word level embedding extraction requires a fast tokenizer.

## 🔮 Future Improvements

- **Inplace Quantization**: Script for quantization of models for AWQ and GPTQ.
- **New quantization methods**: Extend support for new quantization methods.
- **Attention Visualization**: Include tools to visualize attention patterns alongside complexity predictions.