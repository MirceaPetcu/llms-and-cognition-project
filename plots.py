import json
import matplotlib.pyplot as plt

# Load the first JSON file
with open('results/last_embeddings_sentence_ridge_l1_bold_response_LH_float16_sentence_Qwen2.5-7B-Instruct_0_999.json', 'r') as f:
    dict1 = json.load(f)

# Load the second JSON file
with open('results/last_embeddings_sentence_ridge_l1_bold_response_LH_gptq_sentence_Qwen2.5-7B-Instruct-GPTQ-Int4_0_999.json', 'r') as f:
    dict2 = json.load(f)

# Extract x-axis values (layer numbers) and y-axis values (pearson correlations) for dict1
# Convert keys from strings to integers and sort them to ensure proper order
layers1 = sorted([int(k) for k in dict1.keys()])
pearson1 = [dict1[str(layer)]["pearson"] for layer in layers1]

# Do the same for dict2
layers2 = sorted([int(k) for k in dict2.keys()])
pearson2 = [dict2[str(layer)]["pearson"] for layer in layers2]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(layers1, pearson1, marker='o', linestyle='-', label='Float32 Model')
plt.plot(layers2, pearson2, marker='s', linestyle='--', label='GPTQ-Int4 Model')

# Set the axis labels and title
plt.xlabel('Layer Number')
plt.ylabel('Pearson Correlation')
plt.title('Pearson Correlation vs Layer Number for Two Models')

# Add a legend and grid
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
