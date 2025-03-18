import os
import pickle
import logging
from datetime import datetime
import pandas as pd
import json
from arguments import PreprocessingArguments


def setup_logger(file_name: str = 'script.log') -> logging.Logger:
    logger = logging.getLogger("script_logger")
    logger.setLevel(logging.INFO)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = file_name + f"_{timestamp}.log"
    os.makedirs('logs', exist_ok=True)
    handler = logging.FileHandler(os.path.join('logs', file_name),
                                  mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def prepare_input(input: pd.Series,
                  args: PreprocessingArguments,
                  logger: logging.Logger = None) -> tuple[str, list[any]]:
    if args.data.split('.')[-1] in ('csv', 'tsv'):
        text = args.additional_prompt + input[args.text_column] if args.additional_prompt else input[args.text_column]
        targets = input[args.target_column]
        return text, targets
    else:
        logger.error(f"Dataset type not supported")
        raise ValueError(f"Dataset type not supported")


def prepare_sample(text: str, targets: any, entry: pd.Series, args: PreprocessingArguments) -> dict:
    sample = {'text': text, 'targets': targets, 'id': entry[args.id_column]}
    if args.word_embedding:
        sample['word'] = entry[args.word_column]
    sample['lang'] = entry[args.lang_column] if args.lang_column else None

    return sample


def get_output_dir(args: PreprocessingArguments, logger: logging.Logger) -> str:
    output_dir_name = f"{args.data_keyword}_{args.dtype}"
    os.makedirs(output_dir_name, exist_ok=True)
    logger.info(f"Output directory created: {output_dir_name}")
    return output_dir_name


def save_processed_dataset(output_dir_name: str, model_id: str, data: list[dict],
                           logger: logging.Logger) -> None:
    output_file_name = f"{model_id.split('/')[1]}.pkl"
    try:
        with open(os.path.join(output_dir_name, output_file_name), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Processed dataset saved: {output_file_name}")
    except Exception as e:
        logger.error(f"Error saving the processed dataset: {e}")


def free_kaggle_disk_space(logger: logging.Logger) -> None:
    try:
        os.system('rm -rf /root/.cache/huggingface/hub/*')
        logger.info("Kaggle disk space freed")
    except Exception as e:
        logger.error(f"Error freeing Kaggle disk space: {e}")


def get_processed_dataset(data_path: str):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_preduction_results(results: dict, path: str) -> None:
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', path), 'w') as f:
        json.dump(results, f, indent=4)


def load_config(config_path):
    """Load configuration from a specified JSON file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            return json.load(file)
    raise FileNotFoundError(f"Config file {config_path} not found!")