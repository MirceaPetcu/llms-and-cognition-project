import pandas as pd
from transformers import set_seed
from tqdm import tqdm
from huggingface_hub import login
from model import Model
from utils import (prepare_input,
                   prepare_sample,
                   get_output_dir,
                   save_processed_dataset,
                   setup_logger,
                   free_kaggle_disk_space,
                   load_config)
import argparse
import json
from arguments import PreprocessingArguments


def parse_args():
    """Parse command-line arguments and load configuration from a JSON file if provided."""
    parser = argparse.ArgumentParser(description="Load config from JSON or CLI arguments.")
    parser.add_argument('--hf_token', type=str,
                        default='', required=False,
                        help='Huggingface token')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help='Model ID')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float32', 'float16', 'gptq', 'bnb', 'awq'],
                        help='Quantization type')
    parser.add_argument('--dataset_config_file', type=str, default='mlsp.json',
                        help='Path to dataset configuration JSON file')
    parser.add_argument('--last_embedding', type=bool, default=True, help='Whether to store last embedding or not')
    parser.add_argument('--mean_embedding', type=bool, default=True, help='Whether to store mean embedding or not')
    parser.add_argument('--word_embedding', type=bool, default=True, help='Whether to store word embedding or not')
    parser.add_argument('--additional_prompt', type=str, default="", help='Additional prompt')
    return parser.parse_args()


def main(args: PreprocessingArguments):
    login(args.hf_token)
    set_seed(6)
    logger = setup_logger(f"{args.data_keyword}_{args.model.split('/')[1]}_{args.dtype}.log")
    if args.data.split('.')[-1] in ('tsv'):
        df = pd.read_csv(args.data, sep="\t")
    elif args.data.split('.')[-1] in ('csv'):
        df = pd.read_csv(args.data)
    else:
        raise ValueError("Only tsv and csv files are supported")
    output_dir_name = get_output_dir(args, logger)
    logger.info("Starting processing")
    logger.info(f"Processing model {args.model}")
    model = Model(weights_dtype=args.dtype,
                  model_id=args.model,
                  last_embedding=args.last_embedding,
                  mean_embedding=args.mean_embedding,
                  word_embedding=args.word_embedding,
                  logger=logger)
    logger.info(f"Model {args.model} loaded successfully")
    processed_dataset = []
    not_assigned_words = 0
    total_samples = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text, targets = prepare_input(input=row,
                                      args=args,
                                      logger=logger)
        sample = prepare_sample(text=text,
                                targets=targets,
                                entry=row,
                                args=args)
        outputs, nth_tokens = model.inference(text=text,
                                              target_word=sample.get('word', None))
        sample['nth_tokens'] = nth_tokens if nth_tokens is not None else None
        if args.word_embedding and nth_tokens is None:
            sample['nth_tokens'] = (-2,-1)
            not_assigned_words += 1
        sample = model.process_output_embeddings(outputs, sample)
        processed_dataset.append(sample)
        total_samples += 1
    save_processed_dataset(output_dir_name, args.model, processed_dataset, logger)
    logger.info(f"Processing completed for model {args.model}")
    print(f"Number of not assigned words: {not_assigned_words}")
    print(f"Total samples {total_samples}")

if __name__ == '__main__':
    args = parse_args()
    try:
        dataset_config = load_config(args.dataset_config_file)
        preprocessing_args = PreprocessingArguments(
            hf_token=args.hf_token,
            model=args.model,
            dtype=args.dtype,
            data=dataset_config.get('dataset_path', ''),
            additional_prompt=args.additional_prompt,
            data_keyword=dataset_config.get('data_keyword', ''),
            text_column=dataset_config.get('text_column', ''),
            target_column=dataset_config.get('target_column', []),
            lang_column=dataset_config.get('lang_column', ''),
            id_column=dataset_config.get('id_column', ''),
            word_column=dataset_config.get('word_column', ''),
            last_embedding=args.last_embedding,
            mean_embedding=args.mean_embedding,
            word_embedding=args.word_embedding,
            dataset_config_file=args.dataset_config_file,
        )
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config file: {str(e)}")
        exit(1)

    
    main(preprocessing_args)
