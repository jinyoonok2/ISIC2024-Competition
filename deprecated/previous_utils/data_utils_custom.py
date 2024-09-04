import os
import shutil
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import numpy as np
import torch

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

def augment_and_duplicate_images(image_paths, target_dir, multiplier=4):
    augment_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    ])

    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        original_img_path = os.path.join(target_dir, f"{img_name}.jpg")
        shutil.copy(img_path, original_img_path)
        # print(f"Copied original image: {original_img_path}")

        for i in range(1, multiplier + 1):
            augmented_img = augment_transform(img)
            augmented_img = transforms.ToPILImage()(augmented_img)  # Convert back to PIL image
            augmented_img_path = os.path.join(target_dir, f"{img_name}_augmented{i}.jpg")
            augmented_img.save(augmented_img_path)
            # print(f"Saved augmented image: {augmented_img_path}")

    print(f"Augmented and duplicated {len(image_paths)} images with a multiplier of {multiplier}.")

def copy_images(df, base_dir):
    for _, row in df.iterrows():
        img_path = row['image_path']
        target_class = str(int(row['target']))  # Convert the target to an integer and then to a string
        dest_dir = os.path.join(base_dir, target_class)

        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}")
            continue

        shutil.copy(img_path, dest_dir)
        # print(f"Copied image {img_path} to {dest_dir}")

def create_train_valid_folders_custom(config):
    # Initialize the base directory depending on the dataset mode
    base_dir = os.path.abspath(f"./datasets/{config['dataset_mode']}")
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')

    # Check if the train and valid directories already exist
    if os.path.exists(train_dir) and os.path.exists(valid_dir):
        print(f"Train and validation directories already exist. Skipping dataset creation.")
        return train_dir, valid_dir

    # Load 2024 dataset metadata
    isic2024_path = os.path.abspath('./datasets/isic-2024-challenge/train-metadata.csv')
    df_2024 = pd.read_csv(isic2024_path, usecols=['isic_id', 'target'])
    df_2024['image_path'] = './datasets/isic-2024-challenge/train-image/image/' + df_2024['isic_id'] + '.jpg'

    # Convert image paths to absolute paths
    df_2024['image_path'] = df_2024['image_path'].apply(lambda x: os.path.abspath(x))

    # Print directory information
    print(f"Base directory: {base_dir}")
    print(f"Train directory: {train_dir}")
    print(f"Valid directory: {valid_dir}")

    # Handling "practice" mode
    if config['dataset_mode'] in ['2024prac_custom', 'combinedprac_custom']:
        df_target_0 = df_2024[df_2024['target'] == 0]
        df_target_1 = df_2024[df_2024['target'] == 1]

        df_target_0_sampled = df_target_0.sample(frac=0.2, random_state=config['seed']).reset_index(drop=True)
        df_2024 = pd.concat([df_target_0_sampled, df_target_1], ignore_index=True)

    # Handling "combined" modes
    if config['dataset_mode'] in ['combined_custom', 'combinedprac_custom']:
        # Load 2019 and 2020 datasets
        isic2019_path = os.path.abspath('./datasets/isic-2019-dataset-resized-256/train-metadata.csv')
        isic2020_path = os.path.abspath('./datasets/isic-2020-dataset-resized-256/train-metadata.csv')
        df_2019 = pd.read_csv(isic2019_path, usecols=['isic_id', 'target'])
        df_2020 = pd.read_csv(isic2020_path, usecols=['isic_id', 'target'])

        # Filter malignant data (target == 1) for 2019 and 2020
        df_2019_malignant = df_2019[df_2019['target'] == 1].copy()
        df_2020_malignant = df_2020[df_2020['target'] == 1].copy()

        # Add columns for image path and years
        df_2019_malignant['image_path'] = './datasets/isic-2019-dataset-resized-256/train-image/image/' + df_2019_malignant['isic_id'] + '.jpg'
        df_2020_malignant['image_path'] = './datasets/isic-2020-dataset-resized-256/train-image/image/' + df_2020_malignant['isic_id'] + '.jpg'
        df_2019_malignant['image_path'] = df_2019_malignant['image_path'].apply(lambda x: os.path.abspath(x))
        df_2020_malignant['image_path'] = df_2020_malignant['image_path'].apply(lambda x: os.path.abspath(x))

        # Combine datasets
        df_combined = pd.concat([df_2024, df_2019_malignant, df_2020_malignant], ignore_index=True)
    else:
        df_combined = df_2024

    # Split the dataset into train and validation sets
    train_df, valid_df = train_test_split(df_combined, test_size=config['split_ratio'], stratify=df_combined['target'], random_state=config['seed'])

    # Create directories if they don't exist
    for subdir in ['train', 'valid']:
        for class_dir in ['0', '1']:
            path = os.path.join(base_dir, subdir, class_dir)
            os.makedirs(path, exist_ok=True)

    # Copy images to train and valid directories
    copy_images(train_df, train_dir)
    print(f"Copied {len(train_df)} images to train directory.")
    copy_images(valid_df, valid_dir)
    print(f"Copied {len(valid_df)} images to valid directory.")

    # Augment and duplicate class 1 images for training set only if combined mode
    if config['dataset_mode'] in ['combined_custom', 'combinedprac_custom']:
        train_class_1_paths = train_df[train_df['target'] == 1]['image_path'].tolist()
        augment_and_duplicate_images(train_class_1_paths, os.path.join(train_dir, '1'))
    else:
        # Augment and duplicate class 1 images for both training and validation sets
        train_class_1_paths = train_df[train_df['target'] == 1]['image_path'].tolist()
        valid_class_1_paths = valid_df[valid_df['target'] == 1]['image_path'].tolist()

        augment_and_duplicate_images(train_class_1_paths, os.path.join(train_dir, '1'))
        augment_and_duplicate_images(valid_class_1_paths, os.path.join(valid_dir, '1'))

    return train_dir, valid_dir


def get_datasets_based_on_split_option(train_dir, valid_dir, config):
    if config['split_option'] == 'train':
        raise ValueError("The 'train' split option is not allowed for 2024_custom and 2024_custom_practice modes.")

    elif config['split_option'] == 'trainprac':
        train_dataset, valid_dataset = create_imagefolder_datasets(train_dir, valid_dir, config)
        train_loader, valid_loader = create_imagefolder_dataloaders(train_dataset, valid_dataset, config)
        return train_loader, valid_loader

    elif config['split_option'] == 'valid':
        valid_dataset = create_imagefolder_datasets(None, valid_dir, config)
        _, valid_loader = create_imagefolder_dataloaders(None, valid_dataset, config)
        return None, valid_loader

def create_imagefolder_datasets(train_dir, valid_dir, config):
    train_transform = transforms.Compose([
        transforms.Resize(config['input_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor()
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(config['input_size']),
        transforms.ToTensor()
    ])

    train_dataset = ImageFolder(root=train_dir, transform=train_transform) if train_dir else None
    valid_dataset = ImageFolder(root=valid_dir, transform=valid_transform) if valid_dir else None

    if train_dataset:
        print(f"Created ImageFolder training dataset with {len(train_dataset)} samples.")
    if valid_dataset:
        print(f"Created ImageFolder validation dataset with {len(valid_dataset)} samples.")

    return train_dataset, valid_dataset

def create_imagefolder_dataloaders(train_dataset, valid_dataset, config):
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True) if train_dataset else None
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False) if valid_dataset else None

    print(f"Created DataLoaders with batch size {config['batch_size']}.")

    return train_loader, valid_loader
