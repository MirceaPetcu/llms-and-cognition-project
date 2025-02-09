import json
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import argparse


def parse_args():
    parser = argparse.ArgumentParser('Plotting')
    parser.add_argument('--model', default='Qwen2.5-7B-Instruct',
                        type=str, required=False, help='Model name')
    parser.add_argument('--quant', default='float16',
                        type=str, required=False, help='Quantization type')
    parser.add_argument('--group_by', default='quant',
                        type=str, required=False, help='Group by')
    return parser.parse_args()


def parse_results(args):
    results = os.listdir('results')
    results_by_configuration = defaultdict(dict)
    plot_title = ''
    for filename in results:
        components = filename.split('_')
        model = components[10]
        quant = components[8]
        regressor = components[3]
        normalization = components[4]
        embedding = components[:2]
        if args.group_by == 'quant' and args.quant in quant:
            with open(os.path.join('results', filename), 'r') as f:
                data = json.load(f)
            key = (f"{model}").replace('\n', '').strip()
            results_by_configuration[key] = data
            plot_title = f"{embedding[0]} {embedding[1]} {regressor} {normalization} {args.quant}"
        if args.group_by == 'model' and args.model in model:
            with open(os.path.join('results', filename), 'r') as f:
                data = json.load(f)
            key = (f"{quant+' ' + model[model.find(args.model):]}").replace('\n', '').strip()
            results_by_configuration[key] = data
            plot_title = f"{embedding[0]} {embedding[1]} {regressor} {normalization} {args.model}"

    return results_by_configuration, plot_title


def main(results, model, plot_title):
    plt.figure(figsize=(22, 12))

    markers = ['o', 'x', 's', 'D', '^', 'v', '<', '>']
    linestyles = ['-', '--', '-.', ':']
    for i, (title, data) in enumerate(results.items()):

        layers = sorted([int(k) for k in data.keys()])
        pearson = [data[str(layer)]["pearson"] for layer in layers]

        plt.plot(layers, pearson, marker=markers[i%7], linestyle=linestyles[i%4], label=title)


    plt.xlabel('Layer Number')
    plt.ylabel('Pearson Correlation')
    plt.title(f'Pearson Correlation vs Layer Number for {plot_title}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{plot_title}.png')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs('plots', exist_ok=True)
    results, plot_title = parse_results(args)
    main(results, args.model, plot_title)