import os
import pickle
import argparse
import logging
from typing import List, Any, Tuple, Union
from datetime import datetime
import pandas as pd
from nltk.tokenize import word_tokenize
import yaml
import re
import json


def setup_logger(file_name: str = 'script.log') -> logging.Logger:
    logger = logging.getLogger("script_logger")
    logger.setLevel(logging.INFO)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = file_name + f"_{timestamp}.log"
    os.makedirs('logs', exist_ok=True)
    handler = logging.FileHandler(os.path.join('logs', file_name))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def prepare_input(input, args: argparse.Namespace, logger: logging.Logger = None) -> Tuple[str, List[any]]:
    if args.data.split('.')[-1] in ('csv', 'tsv'):
        text = args.additional_prompt + input[args.text_column] if args.additional_prompt else input[args.text_column]
        targets = input[args.target_column]
        return text, targets
    else:
        logger.error(f"Dataset type not supported")
        raise ValueError(f"Dataset type not supported")


def prepare_sample(text: str, targets: Any, entry: pd.Series, args: argparse.Namespace) -> dict:
    sample = {'text': text, 'targets': targets, 'id': entry[args.id_column]}
    regex_patterns = {'english': r"\w+['’]?\w*|[^\w\s]", 'french': r"\w+|[^\w\s]", 'german': r"\w+['’]?\w*|[^\w\s]",
                      'spanish':  r"\w+['’]?\w*|[^\w\s]", 'italian': r"\w+['’]?\w*|[^\w\s]", 'dutch': r"\w+['’]?\w*|[^\w\s]",}
    if args.task == 'word':
        sample['word'] = entry[args.word_column]
        if args.lang_column == 'english':
            words = word_tokenize(entry[args.text_column])
        else:
            words = re.findall(r'\w+(?:-\w+)*|[^\w\s]', entry[args.text_column].lower())
        sample['nth_word'] = words.index(entry[args.word_column].lower())
    sample['lang'] = entry[args.lang_column] if args.lang_column else None

    return sample


def get_output_dir(args: argparse.Namespace, logger: logging.Logger) -> str:
    output_dir_name = f"{args.data_keyword}_{args.dtype}_{args.task}"
    os.makedirs(output_dir_name, exist_ok=True)
    logger.info(f"Output directory created: {output_dir_name}")
    return output_dir_name


def save_processed_dataset(output_dir_name: str, model_id: str, data: List[dict], range_data: Tuple, 
                           logger: logging.Logger) -> None:
    output_file_name = f"{model_id.split('/')[1]}_{range_data[0]}_{range_data[1]}.pkl"
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
