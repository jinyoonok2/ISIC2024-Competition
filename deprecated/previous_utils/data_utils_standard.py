import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import numpy as np
from torch.utils.data import WeightedRandomSampler
from PIL import Image


class ISICDataset(Dataset):
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.iloc[idx, self.metadata.columns.get_loc('image_path')]
        image = Image.open(img_name).convert('RGB')
        label = int(self.metadata.iloc[idx, self.metadata.columns.get_loc('target')])

        if self.transform:
            image = self.transform(image)

        return image, label

def split_dataset_standard(config):
    # Load 2024 dataset metadata
    isic2024_path = './datasets/isic-2024-challenge/train-metadata.csv'
    df_2024 = pd.read_csv(isic2024_path, usecols=['isic_id', 'target'])
    df_2024['image_path'] = './datasets/isic-2024-challenge/train-image/image/' + df_2024['isic_id'] + '.jpg'

    # Handling "practice" mode
    if config['dataset_mode'] in ['2024prac', 'combinedprac']:
        df_target_0 = df_2024[df_2024['target'] == 0]
        df_target_1 = df_2024[df_2024['target'] == 1]

        df_target_0_sampled = df_target_0.sample(frac=0.2, random_state=config['seed']).reset_index(drop=True)
        df_2024 = pd.concat([df_target_0_sampled, df_target_1], ignore_index=True)

    # Handling "combined" modes
    if config['dataset_mode'] in ['combined', 'combinedprac']:
        # Load 2019 and 2020 datasets
        isic2019_path = './datasets/isic-2019-dataset-resized-256/train-metadata.csv'
        isic2020_path = './datasets/isic-2020-dataset-resized-256/train-metadata.csv'
        df_2019 = pd.read_csv(isic2019_path, usecols=['isic_id', 'target'])
        df_2020 = pd.read_csv(isic2020_path, usecols=['isic_id', 'target'])

        # Filter malignant data (target == 1) for 2019 and 2020
        df_2019_malignant = df_2019[df_2019['target'] == 1].copy()
        df_2020_malignant = df_2020[df_2020['target'] == 1].copy()

        # Add columns for image path and year
        df_2019_malignant['image_path'] = './datasets/isic-2019-dataset-resized-256/train-image/image/' + df_2019_malignant['isic_id'] + '.jpg'
        df_2020_malignant['image_path'] = './datasets/isic-2020-dataset-resized-256/train-image/image/' + df_2020_malignant['isic_id'] + '.jpg'

        # Combine datasets
        df_combined = pd.concat([df_2024, df_2019_malignant, df_2020_malignant], ignore_index=True)

        # Use df_combined for further processing
        df_2024 = df_combined

    # Split the dataset into train and validation sets
    train_df, valid_df = train_test_split(df_2024, test_size=config['split_ratio'], stratify=df_2024['target'], random_state=config['seed'])

    return train_df, valid_df

def process_standard_datasets(config):
    df_processed, df_full = load_and_preprocess_data(config)
    train_dataset, valid_dataset, valid_indices = create_datasets(config, df_processed)
    train_loader, valid_loader = create_dataloaders(train_dataset, valid_dataset, config)

    # Print dataset size and class distribution
    if train_loader:
        print(f"Training dataset size: {len(train_loader.dataset)}")

    if valid_loader:
        print(f"Validation dataset size: {len(valid_loader.dataset)}")

    return train_loader, valid_loader

def load_and_preprocess_data(config):
    # Load 2024 metadata
    isic2024_path = './datasets/isic-2024-challenge/train-metadata.csv'
    df_2024 = pd.read_csv(isic2024_path, usecols=['isic_id', 'target'])

    # Add columns for image path and year
    df_2024['image_path'] = './datasets/isic-2024-challenge/train-image/image/' + df_2024['isic_id'] + '.jpg'
    df_2024['year'] = 2024

    # Initialize the combined dataset
    df_combined = df_2024.copy()

    if config['dataset_mode'] in ['combined', 'combinedprac']:
        # Load 2019 and 2020 metadata
        isic2019_path = './datasets/isic-2019-dataset-resized-256/train-metadata.csv'
        isic2020_path = './datasets/isic-2020-dataset-resized-256/train-metadata.csv'
        df_2019 = pd.read_csv(isic2019_path, usecols=['isic_id', 'target'])
        df_2020 = pd.read_csv(isic2020_path, usecols=['isic_id', 'target'])

        # Filter malignant data (target == 1)
        df_2019_malignant = df_2019[df_2019['target'] == 1].copy()
        df_2020_malignant = df_2020[df_2020['target'] == 1].copy()

        # Add columns for image path and year
        df_2019_malignant['image_path'] = './datasets/isic-2019-dataset-resized-256/train-image/image/' + df_2019_malignant['isic_id'] + '.jpg'
        df_2020_malignant['image_path'] = './datasets/isic-2020-dataset-resized-256/train-image/image/' + df_2020_malignant['isic_id'] + '.jpg'
        df_2019_malignant['year'] = 2019
        df_2020_malignant['year'] = 2020

        # Combine malignant data with 2024 data
        df_combined = pd.concat([df_2024, df_2019_malignant, df_2020_malignant], ignore_index=True)

    if config['dataset_mode'] in ['combinedprac', '2024prac']:
        # Apply practice subset only to target == 0 class
        df_target_0 = df_combined[df_combined['target'] == 0]
        df_target_1 = df_combined[df_combined['target'] == 1]

        # Sample the target 0 class
        df_target_0_sampled = df_target_0.sample(frac=0.2, random_state=config['seed']).reset_index(drop=True)

        # Combine sampled target 0 with all target 1 data
        df_combined_practice = pd.concat([df_target_0_sampled, df_target_1], ignore_index=True)

        return df_combined_practice, df_2024

    elif config['dataset_mode'] == '2024':
        # Only use the 2024 dataset
        return df_2024, df_2024

    else:
        return df_combined, df_2024

def create_datasets(config, df_processed):
    augmentation_option = config.get('augmentation', 'default')

    if augmentation_option == 'default':
        train_transform = transforms.Compose([
            transforms.Resize(config['input_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor()
        ])
    elif augmentation_option == 'strong':
        train_transform = transforms.Compose([
            transforms.Resize(config['input_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor()
        ])

    print(f"Using '{augmentation_option}' augmentation.")

    valid_transform = transforms.Compose([
        transforms.Resize(config['input_size']),
        transforms.ToTensor()
    ])

    full_dataset = ISICDataset(df_processed, transform=None)
    print(f"Current split option: {config['split_option']}")

    if config['split_option'] == 'train':
        full_dataset.transform = train_transform
        train_labels = df_processed['target'].value_counts()
        print(f"Training dataset class distribution: {train_labels.to_dict()}")
        return full_dataset, None, None
    else:
        train_indices, valid_indices = train_test_split(
            range(len(full_dataset)),
            test_size=config['split_ratio'],
            stratify=df_processed['target'],
            random_state=config['seed']
        )

        train_dataset = Subset(full_dataset, train_indices)
        valid_dataset = Subset(full_dataset, valid_indices)

        train_dataset.dataset.transform = train_transform
        valid_dataset.dataset.transform = valid_transform

        train_labels = df_processed.iloc[train_indices]['target'].value_counts()
        valid_labels = df_processed.iloc[valid_indices]['target'].value_counts()

        print(f"Training dataset class distribution: {train_labels.to_dict()}")
        print(f"Validation dataset class distribution: {valid_labels.to_dict()}")

        return train_dataset, valid_dataset, valid_indices

def create_dataloaders(train_dataset, valid_dataset, config):
    if train_dataset is not None:
        sampling_option = config.get('sampling', 'default')

        if sampling_option == 'weighted':
            # Calculate the class counts (number of samples per class)
            target_counts = np.bincount(train_dataset.dataset.metadata['target'])

            # Use the scaling factor from the config to control the degree of up-weighting the minority class
            scaling_factor = config.get('scaling_factor', 1)

            # Calculate class weights:
            # Higher scaling_factor will decrease the power value, leading to higher weights for the minority class.
            # Conversely, a lower scaling_factor will result in lower weights for the minority class.
            class_weights = 1. / np.power(target_counts, scaling_factor)

            # Assign a weight to each sample based on its class
            # Samples belonging to minority classes will receive higher weights if the scaling_factor is high
            sample_weights = [class_weights[int(train_dataset.dataset.metadata.iloc[idx]['target'])] for idx in
                              train_dataset.indices]

            # Create a sampler with these weights
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            print("Using Weighted Sampler.")
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler)

        elif sampling_option == 'default':
            print("Using normal distribution (no weighted sampling).")
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

        print(f"Using '{sampling_option}' sampling.")
        print(f"Training loader created with {len(train_loader.dataset)} samples.")
    else:
        train_loader = None

    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
        print(f"Validation loader created with {len(valid_loader.dataset)} samples.")
    else:
        valid_loader = None

    return train_loader, valid_loader
