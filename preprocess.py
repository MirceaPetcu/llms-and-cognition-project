import pandas as pd
from transformers import set_seed
from tqdm import tqdm
import os
from huggingface_hub import login
import argparse
from model import Model
from utils import prepare_input, prepare_sample, get_output_dir, save_processed_dataset, setup_logger, free_kaggle_disk_space
from typing import List
import gc
import argparse
import json
import os

def load_config(config_path):
    """Load configuration from a specified JSON file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            return json.load(file)
    raise FileNotFoundError(f"Config file {config_path} not found!")

def parse_args():
    """Parse command-line arguments and load configuration from a JSON file if provided."""
    parser = argparse.ArgumentParser(description="Load config from JSON or CLI arguments.")
    
    # Add an argument to specify the configuration file
    parser.add_argument('--config', type=str, default="config_fp32.json", help="Path to JSON config file")
    parser.add_argument('--hf_token', type=str, default='', help='Huggingface token')
    parser.add_argument('--model', type=str, default='', help='Model ID')
    
    # Parse known arguments to get the config path first
    args, unknown = parser.parse_known_args()
    
    # Load configuration from the specified JSON file
    config = load_config(args.config)

    # Add the rest of the arguments using config defaults
    parser.add_argument('--data', type=str, default=config.get("data"), help='Data file/data link')
    parser.add_argument('--data_keyword', type=str, default=config.get("data_keyword"), help='Data keyword')
    parser.add_argument('--text_column', type=str, default=config.get("text_column"), help='Text column')
    parser.add_argument('--target_column', type=str, default=config.get("target_column"), help='Target column')
    parser.add_argument('--lang_column', type=str, default=config.get("lang_column"), help='Language column')
    parser.add_argument('--id_column', type=str, default=config.get("id_column"), help='ID column')
    parser.add_argument('--word_column', type=str, default=config.get("word_column"), help='Word column')
    parser.add_argument('--dtype', type=str, default=config.get("dtype"), help='Quantization type')
    parser.add_argument('--inference_type', type=str, default=config.get("inference_type", "forward"), help='Inference type')
    parser.add_argument('--task', type=str, default=config.get("task"), choices=['sentence', 'word'], help='Sentence or word task')
    parser.add_argument('--additional_prompt', type=str, default=config.get("additional_prompt", ""), help='Additional prompt')
    parser.add_argument('--gguf_filename', type=str, default=config.get("gguf_filename", ""), help='GGUF filename to load from')

    return parser.parse_args()



def main(args: argparse.Namespace):
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    os.environ["HUGGINGFACE_TOKEN"] = args.hf_token
    login(args.hf_token)
    set_seed(6)
    logger = setup_logger(f"{args.data_keyword}_{args.model.split('/')[1]}_{args.dtype}_{args.inference_type}_{args.task}.log")
    if args.data.split('.')[-1] in ('tsv'):
        df = pd.read_csv(args.data, sep="\t")
    else:
        df = pd.read_csv(os.path.join('data', args.data))
    output_dir_name = get_output_dir(args, logger)
    logger.info("Starting processing")
    logger.info(f"Processing model {args.model}")
    model = Model(args.dtype, args.inference_type, args.model, args.task, args.gguf_filename, logger)
    logger.info(f"Model {args.model} loaded successfully")
    processed_dataset = []
    range_data =  (0, 1000)
    x = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if i >= range_data[1]:
            save_processed_dataset(output_dir_name, args.model, processed_dataset, range_data, logger)
            range_data = (range_data[1], range_data[1]+1000)
            del processed_dataset
            processed_dataset = []
            gc.collect()
        text, targets = prepare_input(row, args=args, logger=logger)
        sample = prepare_sample(text, targets, row, args=args)
        outputs, nth_tokens = model.inference(text, word_position=sample.get('nth_word', None))
        sample['nth_tokens'] = nth_tokens if nth_tokens is not None else None
        if args.task == 'word' and nth_tokens is None:
            sample['nth_tokens'] = (max(0, sample.get('nth_word', None)-2), sample.get('nth_word', None))
            x += 1
        sample = model.process_output_embeddings(outputs, sample)
        processed_dataset.append(sample)
    # model.free_memory()
    print(x)
    # free_kaggle_disk_space(logger)
    save_processed_dataset(output_dir_name, args.model, processed_dataset, (range_data[0], i), logger)
    logger.info(f"Processing completed for model {args.model}")
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
