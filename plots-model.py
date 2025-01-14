import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

class PearsonCorrelationAnalyzer:
    def __init__(self, results_dir: str, model_name: str, languages: List[str], tasks: List[str]) -> None:
        self.results_dir = results_dir
        self.model_name = model_name
        self.languages = languages
        self.tasks = tasks
        self.pearson_data = {lang: {task: [] for task in tasks} for lang in languages}

    def process_files(self) -> None:
        """Process all YAML files in the given directory."""
        for filename in os.listdir(self.results_dir):
            if filename.endswith(".yaml"):
                self._process_file(filename)

    def _process_file(self, filename: str) -> None:
        """Process each individual YAML file."""
        file_path = os.path.join(self.results_dir, filename)

        lang = next((l for l in self.languages if l in filename), None)
        task = next((t for t in self.tasks if t in filename), None)

        if not lang or not task:
            return

        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        if task == 'sentence':
            self._process_sentence_task(lang, data)
        else:
            self._process_word_task(lang, data)

    def _process_sentence_task(self, lang: str, data: Dict) -> None:
        """Process data for the sentence task."""
        for key, value in data.items():
            normalized_key = key.strip().replace(":", "").lower()
            if 'pearson' in normalized_key and value is not None:
                self.pearson_data[lang]['sentence'].append(value)

    def _process_word_task(self, lang: str, data: Dict) -> None:
        """Process data for the word task."""
        if 'pearson' in data and data['pearson'] is not None:
            self.pearson_data[lang]['word'].append(data['pearson'])

    def calculate_overall_average(self) -> List[float]:
        """Calculate the overall average Pearson correlation for all tasks and languages."""
        overall_avg = []
        num_layers_word = len(self.pearson_data['english']['word'])
        num_layers_sentence = len(self.pearson_data['english']['sentence']) // 2

        for layer in range(max(num_layers_word, num_layers_sentence)):
            word_avg = 0
            sentence_avg = 0

            if layer < num_layers_word:
                word_values = [self.pearson_data[lang]['word'][layer] for lang in self.languages]
                word_avg = np.mean(word_values)

            if layer < num_layers_sentence:
                mean_embedding = self.pearson_data['english']['sentence'][layer]
                last_embedding = self.pearson_data['english']['sentence'][layer + num_layers_sentence]
                sentence_avg = (mean_embedding + last_embedding) / 2

            overall_avg.append((word_avg + sentence_avg) / 2)

        return overall_avg

    def plot_overall_performance(self, overall_avg: List[float]) -> None:
        """Save the overall performance plot."""
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(overall_avg)), overall_avg, label='Overall Performance (Avg)', marker='o', color='b', linestyle='-')
        plt.xlabel('Layer')
        plt.ylabel('Average Pearson Correlation')
        plt.title(f'Overall Model Performance for {self.model_name} (Word and Sentence Tasks Combined)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig('overall_performance.png')
        plt.close()

    def plot_task_performance(self, task: str) -> None:
        """Save the plot for a specific task (word or sentence)."""
        color_map = {'english': 'b', 'french': 'r', 'german': 'm', 'spanish': 'k'}
        plt.figure(figsize=(12, 6))

        for lang in self.languages:
            pearson_values = self.pearson_data[lang][task]
            if not pearson_values:
                print(f"No data for '{task}' task in {lang}")
                continue

            label = f'{lang} - {task}'
            plt.plot(np.arange(len(pearson_values)), pearson_values, label=label, marker='o', color=color_map[lang])

        plt.xlabel('Layer')
        plt.ylabel('Pearson Correlation')
        plt.title(f'Pearson Correlation for {self.model_name} - {task.capitalize()} Task')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(f'{task}_performance.png')
        plt.close()

    def plot_sentence_embeddings(self) -> None:
        """Save the plot for sentence embeddings."""
        color_map = {'english': 'b', 'french': 'r', 'german': 'm', 'spanish': 'k'}
        plt.figure(figsize=(12, 6))

        for lang in self.languages:
            pearson_values = self.pearson_data[lang]['sentence']
            if not pearson_values:
                print(f"No data for 'sentence' task in {lang}")
                continue

            num_layers = len(pearson_values) // 2
            mean_embeddings = pearson_values[:num_layers]
            last_embeddings = pearson_values[num_layers:]

            avg_embeddings = [(mean + last) / 2 for mean, last in zip(mean_embeddings, last_embeddings)]

            plt.plot(np.arange(num_layers), avg_embeddings, label=f'{lang} - sentence (avg)', marker='s', color=color_map[lang], linestyle=':')

        plt.xlabel('Layer')
        plt.ylabel('Pearson Correlation')
        plt.title(f'Pearson Correlation for {self.model_name} - Sentence Task')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig('sentence_embeddings.png')
        plt.close()

    def plot_word_and_sentence_task_avg(self) -> None:
        """Plot and save the overall performance for both word and sentence tasks (averaged across languages)."""
        word_task_avg = []
        sentence_task_avg = []
        
        num_layers_word = len(self.pearson_data['english']['word'])
        num_layers_sentence = len(self.pearson_data['english']['sentence']) // 2
        for layer in range(num_layers_word):
            word_values = [self.pearson_data[lang]['word'][layer] for lang in self.languages]
            word_task_avg.append(np.mean(word_values))
        for layer in range(num_layers_sentence):
            sentence_values = []
            for lang in self.languages:
                mean_embedding = self.pearson_data[lang]['sentence'][layer]
                last_embedding = self.pearson_data[lang]['sentence'][layer + num_layers_sentence]
                avg_value = (mean_embedding + last_embedding) / 2
                sentence_values.append(avg_value)
            sentence_task_avg.append(np.mean(sentence_values))
            
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(num_layers_word), word_task_avg, label='Word Task (Avg)', marker='o', color='b', linestyle='-')
        plt.plot(np.arange(num_layers_sentence), sentence_task_avg, label='Sentence Task (Avg)', marker='s', color='r', linestyle=':')
        plt.xlabel('Layer')
        plt.ylabel('Average Pearson Correlation')
        plt.title(f'Overall Performance for {self.model_name} (Word and Sentence Tasks)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig('word_and_sentence_task_avg.png')
        plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Pearson Correlation Analysis")
    parser.add_argument('--results_dir', type=str, required=True, help="Path to the directory containing result YAML files")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model")
    parser.add_argument('--languages', nargs='+', required=True, help="List of languages to analyze")
    parser.add_argument('--tasks', nargs='+', required=True, help="List of tasks to analyze (e.g., word, sentence)")

    args = parser.parse_args()

    analyzer = PearsonCorrelationAnalyzer(
        results_dir=args.results_dir,
        model_name=args.model_name,
        languages=args.languages,
        tasks=args.tasks
    )

    analyzer.process_files()

    overall_avg = analyzer.calculate_overall_average()
    analyzer.plot_overall_performance(overall_avg)

    analyzer.plot_task_performance('word')
    analyzer.plot_sentence_embeddings()
    analyzer.plot_word_and_sentence_task_avg()
