import os
import random
import numpy as np
import pandas as pd
import timm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torch.optim import AdamW
from torch.nn import BCELoss
from tqdm import tqdm
import torch.nn as nn

# Configuration dictionary
config = {
    'batch_size': 64,
    'num_classes': 1,  # Binary classification (single output with logits)
    'learning_rate': 1e-4,
    'num_epochs': 35,
    'seed': 42,
    'model_names': [  # List of models to train
        'selecsls42b.in1k',
        # 'nextvit_small.bd_in1k_384',
        # 'efficientnet_b3.ra2_in1k'
    ]
}

# Seeding function
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

# Dataset Preprocessing (2024 Only)
def load_and_preprocess_data():
    # Load 2024 metadata
    isic2024_path = './datasets/isic-2024-challenge/train-metadata.csv'
    df_2024 = pd.read_csv(isic2024_path)

    # Add a column for the image path based on the 'isic_id'
    df_2024['image_path'] = './datasets/isic-2024-challenge/train-image/image/' + df_2024['isic_id'] + '.jpg'

    # Add a column to indicate the year (for reference or potential further use)
    df_2024['year'] = 2024

    # Return the 2024 dataset
    return df_2024

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[idx]['target']

        if self.transform:
            image = self.transform(image)

        return image, label

# DataLoader Function with Model-Specific Transforms
def get_dataloader(config, dataframe, model_name):
    # Resolve model-specific data config
    base_model = timm.create_model(model_name, pretrained=True, num_classes=config['num_classes'])
    data_config = timm.data.resolve_model_data_config(base_model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    dataset = CustomDataset(dataframe=dataframe, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    return dataloader

# Training Loop
def train_model(model_name, config, train_dataloader):
    print(f"Training model: {model_name}")

    # Create a directory to save model weights for each epoch
    model_dir = os.path.join('./logs', model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Load the model
    base_model = timm.create_model(model_name, pretrained=True, num_classes=config['num_classes'])  # Binary classification

    # Add sigmoid layer
    model = nn.Sequential(
        base_model,
        nn.Sigmoid()  # Adds a sigmoid layer to the model
    )

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function and optimizer
    criterion = BCELoss()  # BCELoss expects probabilities (after sigmoid)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])

    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(
                1)  # Convert labels to float and reshape

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print loss for this epoch
        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {avg_loss:.4f}")

        # Save model weights after every epoch
        model_path = os.path.join(model_dir, f'epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Saved model weights to {model_path}")

    print(f"Finished training model: {model_name}\n")

    return model

# Main Function to Combine Everything
def main():
    # Set the seed
    seeding(config['seed'])

    # Load and preprocess the data
    df_train = load_and_preprocess_data()

    # Train each model
    for model_name in config['model_names']:
        # Create DataLoader with model-specific transforms
        train_dataloader = get_dataloader(config, df_train, model_name)
        model = train_model(model_name, config, train_dataloader)

if __name__ == "__main__":
    main()
