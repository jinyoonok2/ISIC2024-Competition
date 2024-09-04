import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_pauc_comparison(model_dict, output_image_path="pauc_comparison.png"):
    plt.figure(figsize=(10, 6))

    for model_name, csv_file_path in model_dict.items():
        # Read the CSV file
        data = pd.read_csv(csv_file_path)

        # Assuming the CSV has 'Epoch' and 'pAUC' columns
        epochs = data['Epoch']
        pauc_scores = data['pAUC']

        # Plot the pAUC scores over epochs
        plt.plot(epochs, pauc_scores, label=model_name)

    # Set y-axis limits
    plt.ylim(0, 0.2)

    # Add labels and title to the plot
    plt.xlabel('Epoch')
    plt.ylabel('pAUC Score')
    plt.title('pAUC Scores Comparison Across Models')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig(output_image_path)

    # Show the plot
    plt.show()

# Example usage:
model_dict = {
    "2024 Augmented Partial Frozen": r"C:\Jinyoon_Projects\1_ISIC_2024_competition\logs\7_efb31k_simplev2_2024prac_augmented_nofreeze\logs_2024augmentedsimplev2\efficientnet_b3ra2_in1k_simplehead_2024_custom_practice_default_default_lossbce_imgsz128\pauc_scores.csv",
}

plot_pauc_comparison(model_dict, output_image_path="pauc_comparison.png")
