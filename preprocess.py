import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import os
from huggingface_hub import login
import pickle
import argparse
import logging
from model import Model
from utils import prepare_input, prepare_sample, get_output_dir, save_processed_dataset, setup_logger, free_kaggle_disk_space
from typing import List, Any, Tuple, Union


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--hf_token', type=str, default=None, help='Huggingface token')
    parser.add_argument('--model', type=str, default=[],
                        help='Models ID')
    parser.add_argument('--data', type=str, default='bold_response_LH.csv', help='data file')
    parser.add_argument('--dtype', type=str, default='bnb', help='quantization type')
    parser.add_argument('--inference_type', type=str, default='forward', help='inference type')
    parser.add_argument('--task', type=str, default='sentence', choices=['sentence', 'word'], help='sentence or word')
    parser.add_argument('--additional_prompt', type=str, default=None, help='additional prompt')
    return parser.parse_args()



def main(args: argparse.Namespace):
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    logger = setup_logger(f"{args.data}_{args.model.split('/')[1]}_{args.dtype}_{args.inference_type}_{args.task}.log")
    df = pd.read_csv(os.path.join('data', args.data))
    output_dir_name = get_output_dir(args, logger)
    login(args.hf_token)
    logger.info("Starting processing")
    logger.info(f"Processing model {args.model}")
    model = Model(args.dtype, args.inference_type, args.model, args.task, logger)
    logger.info(f"Model {args.model} loaded successfully")
    new_df = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text, targets = prepare_input(row, dataset_type='dataframe', text_column='sentence',
                                            target_columns=['lang_LH_netw'], additional_prompt=args.additional_prompt, logger=logger)
            sample = prepare_sample(text, targets, row['item_id'])
            outputs = model.inference(text)
            sample = model.process_output_embeddings(outputs, sample)
            new_df.append(sample)
        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")
    model.free_memory()
    free_kaggle_disk_space(logger)
    save_processed_dataset(output_dir_name, args.model, new_df, logger)
    logger.info(f"Processing completed for model")


if __name__ == '__main__':
    args = parse_args()
    main(args)
