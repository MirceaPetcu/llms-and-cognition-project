import os
import pickle
import argparse
import logging
from typing import List, Any, Tuple, Union

def setup_logger(file_name: str = 'script.log') -> logging.Logger:
    logger = logging.getLogger("script_logger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join('logs', file_name))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def prepare_input(input, dataset_type: str, text_column: str = 'sentence', target_columns: List[str] = ['lang_LH_netw'],
                  additional_prompt: str = None, logger: logging.Logger = None) -> Tuple[str, List[any]]:
    if dataset_type == 'dataframe':
        text = additional_prompt + input[text_column] if additional_prompt else input[text_column]
        targets = [input[col] for col in target_columns]
        return text, targets
    else:
        logger.error(f"Dataset type {dataset_type} not supported")
        raise ValueError(f"Dataset type {dataset_type} not supported")


def prepare_sample(text: str, targets: Any, id: Union[str,int], word = None) -> dict:
    sample = {'text': text, 'targets': targets, 'id': id}
    if word:
        sample['word'] = word
    return sample


def get_output_dir(args: argparse.Namespace, logger: logging.Logger) -> str:
    output_dir_name = f"{args.data}_{args.model_family}_{args.dtype}_{args.inference_type}_{args.task}"
    os.makedirs(output_dir_name, exist_ok=True)
    logger.info(f"Output directory created: {output_dir_name}")
    return output_dir_name


def save_processed_dataset(output_dir_name: str, model_id: str, data: List[dict], logger: logging.Logger) -> None:
    output_file_name = f"{model_id.split('/')[1]}.pkl"
    try:
        with open(os.path.join(output_dir_name, output_file_name), 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Processed dataset saved: {output_file_name}")
    except Exception as e:
        logger.error(f"Error saving the processed dataset: {e}")
