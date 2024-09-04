import os
import random
import torch
import numpy as np
from collections import Counter
from torchvision.datasets import ImageFolder
from utils.data_utils_custom import create_train_valid_folders_custom, get_datasets_based_on_split_option
from utils.data_utils_standard import split_dataset_standard, process_standard_datasets
from utils.classification_model_utils import train_model, create_model
from utils.classification_evaluation import evaluate_model, visualize_dataloader_batch  # Updated import

# Configuration dictionary
CONFIG = {
    'model_name': 'efficientnet_b3.ra2_in1k',
    'project_name': 'ISIC_2024_Competition',
    'artifact_name': 'isic2024-simplehead-model',
    'learning_rate': 0.0003,
    'batch_size': 64,
    'seed': 42,
    'wandb_log': False,

    # data options
    'dataset_mode': 'combined_custom',  # '2024', '2024prac', '2024_custom', '2024prac_custom', or 'combined', 'combinedprac', 'combined_custom', 'combinedprac_custom'
    'split_ratio': 0.2,  # Ratio for validation split
    'augmentation': 'default',  # Options: 'default', 'strong', etc.
    'sampling': 'default',  # Options: 'default', 'weighted'

    # train options
    'pretrained': True,
    'num_epochs': 20,
    'split_option': 'trainprac',  # 'train', 'trainprac', or 'valid'
    'head_type': 'simpleheadv2',  # # 'simplehead', 'scsa', 'simpleheadv2'
    'valid_model_path': [],
    'input_size': (128, 128),  # Change input size here
    'loss_function': 'bce',  # 'bce' or 'focal'

    # Visualization
    'print_backbone': False,
    'monte_carlo_visual': False,
    'number_of_samples': 5000,

    # Sampler
    'use_weighted_sampler': False,  # Set to True to use the weighted sampler, False for normal distribution
    'scaling_factor': 1,  # Adjust this scaling factor as needed
    'freeze': 'unfreeze', # 'freezeall', 'unfreeze', 'freezecustom1'
}

def seeding(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    print("Seeding done ...")

def print_final_class_distribution(train_dir, valid_dir):
    train_dataset = ImageFolder(root=train_dir)
    valid_dataset = ImageFolder(root=valid_dir)

    train_labels = [sample[1] for sample in train_dataset.samples]
    valid_labels = [sample[1] for sample in valid_dataset.samples]

    train_distribution = Counter(train_labels)
    valid_distribution = Counter(valid_labels)

    print(f"Final training dataset class distribution: {dict(train_distribution)}")
    print(f"Final validation dataset class distribution: {dict(valid_distribution)}")


def main():
    seeding(CONFIG['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model save prefix
    model_save_prefix = f"{CONFIG['model_name'].replace('.', '')}_{CONFIG['head_type']}_{CONFIG['dataset_mode']}_{CONFIG['augmentation']}_{CONFIG['sampling']}_loss{CONFIG['loss_function']}_imgsz{CONFIG['input_size'][0]}"

    experiment_dir = os.path.join("./logs", model_save_prefix)
    plots_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if CONFIG['dataset_mode'] in ['2024_custom', '2024prac_custom', 'combined_custom', 'combinedprac_custom']:
        train_dir, valid_dir = create_train_valid_folders_custom(CONFIG)
        train_loader, valid_loader = get_datasets_based_on_split_option(train_dir, valid_dir, CONFIG)
    else:
        train_df, valid_df = split_dataset_standard(CONFIG)
        train_loader, valid_loader = process_standard_datasets(CONFIG)

    if CONFIG['split_option'] == 'trainprac':
        visualize_dataloader_batch(train_loader, "Training Practice Data Batch")
        model = train_model(train_loader, valid_loader, CONFIG, device, model_save_prefix)

    elif CONFIG['split_option'] == 'valid':
        visualize_dataloader_batch(valid_loader, "Validation Data Batch")
        for model_path in CONFIG['valid_model_path']:
            model = create_model(CONFIG, device, pretrained=False)  # Do not load pretrained weights for evaluation
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)
            print(f"Evaluating model: {model_path}")

            # Evaluate the model for all epochs (assuming single evaluation call, no epoch multiple condition)
            evaluate_model(model, valid_loader, device, epoch=None, save_dir=plots_dir)  # Save evaluation results

if __name__ == "__main__":
    main()
