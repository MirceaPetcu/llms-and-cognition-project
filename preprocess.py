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


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--hf_token', type=str, default=None, help='Huggingface token')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B',
                        help='Models ID')
    parser.add_argument('--data', type=str, default='hf://datasets/MLSP2024/MLSP2024/English/multils_test_english_lcp_labels.tsv',
                 help='data file/ data link')
    parser.add_argument('--data_keyword', type=str, default='multils_test_all_lcp_labels',
                        help='data keyword')
    parser.add_argument('--text_column', type=str, default='context', help='text column')
    parser.add_argument('--target_column', type=str, default=['complexity'], help='target column')
    parser.add_argument('--lang_column', type=str, default=None, help='language column')
    parser.add_argument('--id_column', type=str, default='id', help='id column')
    parser.add_argument('--word_column', type=str, default='target', help='word column')
    parser.add_argument('--dtype', type=str, default='bnb', help='quantization type')
    parser.add_argument('--inference_type', type=str, default='forward', help='inference type')
    parser.add_argument('--task', type=str, default='word', choices=['sentence', 'word'], help='sentence or word')
    parser.add_argument('--additional_prompt', type=str, default=None, help='additional prompt')
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
    model = Model(args.dtype, args.inference_type, args.model, args.task, logger)
    logger.info(f"Model {args.model} loaded successfully")
    processed_dataset = []
    range_data =  (0, 1000)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            if i >= range_data[1]:
                save_processed_dataset(output_dir_name, args.model, processed_dataset, range_data, logger)
                range_data = (range_data[1], range_data[1]+1000)
                del processed_dataset
                processed_dataset = []
                gc.collect()
            text, targets = prepare_input(row, args=args, logger=logger)
            sample = prepare_sample(text, targets, row, args=args)
            outputs = model.inference(text)
            sample = model.process_output_embeddings(outputs, sample)
            processed_dataset.append(sample)
        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")
    model.free_memory()
    free_kaggle_disk_space(logger)
    save_processed_dataset(output_dir_name, args.model, processed_dataset, (range_data[0], i), logger)
    logger.info(f"Processing completed for model")


if __name__ == '__main__':
    args = parse_args()
    main(args)
