import subprocess
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser('Pipeline')
    parser.add_argument('--quant', type=str, default='', help='Path to the configuration file')
    parser.add_argument('--hf_token', type=str, default='', help='Huggingface token')
    parser.add_argument('--model', type=str, default='', help='Model ID')
    return parser.parse_args()

def main(args):
    config_file = f"config_{args.quant}.json"
    hf_token = args.hf_token
    model = args.model
    # conda = ["conda", "run", "-n", "llms_env", ]
    command = ["python", "preprocess.py", "--config", config_file, "--hf_token", hf_token, "--model", model]

    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    
    with open(f"config_{args.quant}.json", "r") as f:
        config = json.load(f)

    dataset = f"{config['data_keyword']}_{args.quant}_{config['task']}/{args.model.split('/')[-1]}_0_999.pkl"
    command = ["python", "cv.py", "--data", dataset]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

if __name__ == '__main__':
    args = parse_args()
    main(args)
