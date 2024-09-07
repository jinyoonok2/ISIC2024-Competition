import os
import pandas as pd
import timm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from utils.lgbm_train_cv import train_lightgbm_model  # Import LightGBM training function

# Configuration dictionary
config = {
    'batch_size': 32,
    'model_path': './logs/logs_effb3ra2in1k/epoch_19.pth',  # Only EfficientNet
    'feature_updated_csv': './datasets/isic-2024-challenge/feature_updated_dataset.csv',
    'feature_engineered_csv': './datasets/isic-2024-challenge/feature_engineered_dataset.csv',
    'post_processed_csv': './datasets/isic-2024-challenge/post_processed_dataset.csv',
    'lightgbm_config': {  # LightGBM configuration stays the same
        'model_name': 'lightgbm',
        'log_dir': './logs/lightgbm',
        'n_splits': 5,
        'seed': 42,
        'display_feature_importance': True,
        'feature_columns': [],
        'target_column': 'target',
        'group_column': 'patient_id',
        'lgb_params': {
            "objective": "binary",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": 200,
            'learning_rate': 0.05,
            'lambda_l1': 0.0004681884533249742,
            'lambda_l2': 8.765240856362274,
            'num_leaves': 136,
            'feature_fraction': 0.5392005444882538,
            'bagging_fraction': 0.9577412548866563,
            'bagging_freq': 6,
            'min_child_samples': 60,
            "device": "gpu"
        },
        'save_best_model': True
    }
}

# Numeric columns
num_cols = [
    'age_approx',
    'clin_size_long_diam_mm',
    'tbp_lv_A', 'tbp_lv_Aext',
    'tbp_lv_B', 'tbp_lv_Bext',
    'tbp_lv_C', 'tbp_lv_Cext',
    'tbp_lv_H', 'tbp_lv_Hext',
    'tbp_lv_L', 'tbp_lv_Lext',
    'tbp_lv_areaMM2',
    'tbp_lv_area_perim_ratio',
    'tbp_lv_color_std_mean',
    'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL',
    'tbp_lv_deltaLBnorm',
    'tbp_lv_eccentricity',
    'tbp_lv_minorAxisMM',
    'tbp_lv_nevi_confidence',
    'tbp_lv_norm_border',
    'tbp_lv_norm_color',
    'tbp_lv_perimeterMM',
    'tbp_lv_radial_color_std_max',
    'tbp_lv_stdL', 'tbp_lv_stdLExt',
    'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
    'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
]

# Non-numeric data found in columns: ['anatom_site_general', 'image_type', 'attribution', 'copyright_license', 'combined_anatomical_site']
cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple", "anatom_site_general"]

# List of columns to drop based on comparison with test-metadata
columns_to_drop = [
    'iddx_3', 'iddx_2', 'iddx_5', 'iddx_full',
    'tbp_lv_dnn_lesion_confidence', 'lesion_id',
    'mel_mitotic_index', 'mel_thick_mm', 'iddx_1', 'iddx_4',
    'image_type', 'attribution', 'copyright_license'
]

# Custom Dataset Class without storing image paths in DataFrame
class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


# DataLoader Function with Model-Specific Transforms
def get_dataloader(image_paths, model_name, batch_size):
    base_model = timm.create_model(model_name, pretrained=True, num_classes=1)
    data_config = timm.data.resolve_model_data_config(base_model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    dataset = CustomDataset(image_paths=image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader


# Function to generate features using the saved model
def generate_features(model_name, model_path, image_paths, df, config):
    print(f"Generating features using model: {model_name}")

    base_model = timm.create_model(model_name, pretrained=True, num_classes=1)

    model = nn.Sequential(
        base_model,
        nn.Sigmoid()  # Sigmoid layer to output probabilities
    )

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = get_dataloader(image_paths, model_name, config['batch_size'])

    predictions = []

    with torch.no_grad():
        for inputs in tqdm(dataloader, desc=f"Processing {model_name}"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy().flatten())

    df[model_name] = predictions

    return df


# Preprocessing Function for Filling Missing Values in Numeric Columns
def fill_na_with_median(df, num_cols):
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


# Categorical Encoding Function
def encode_categorical(df, cat_cols, category_encoder=None):
    if category_encoder is None:
        category_encoder = OrdinalEncoder(
            categories='auto',
            dtype=int,
            handle_unknown='use_encoded_value',
            unknown_value=-2,
            encoded_missing_value=-1,
        )

    X_cat = category_encoder.fit_transform(df[cat_cols])

    for c, cat_col in enumerate(cat_cols):
        df[cat_col] = X_cat[:, c]

    return df, category_encoder


def main():
    if os.path.exists(config['feature_updated_csv']):
        print(f"Feature updated CSV already exists at: {config['feature_updated_csv']}")
        df = pd.read_csv(config['feature_updated_csv'])
    else:
        df = pd.read_csv('./datasets/isic-2024-challenge/train-metadata.csv')

        # Drop unnecessary columns, including those missing in test-metadata
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Step 1: Fill missing values with median in numeric columns
        df = fill_na_with_median(df, num_cols)

        # Generate features using EfficientNet
        image_paths = './datasets/isic-2024-challenge/train-image/image/' + df['isic_id'] + '.jpg'
        df = generate_features('efficientnet_b3.ra2_in1k', config['model_path'], image_paths.tolist(), df, config)

        # Encode categorical columns
        df, category_encoder = encode_categorical(df, cat_cols)

        # Save the updated DataFrame with generated features
        df.to_csv(config['feature_updated_csv'], index=False)
        print(f"Saved DataFrame with model-generated features to {config['feature_updated_csv']}")

    # Check if post-processed CSV already exists
    if os.path.exists(config['post_processed_csv']):
        print(f"Post-processed CSV already exists at: {config['post_processed_csv']}")
        df = pd.read_csv(config['post_processed_csv'])
    else:
        # Post-process empty cells by filling them with -1
        print("Post-processing the DataFrame by filling missing values with -1...")
        df.fillna(-1, inplace=True)

        # Save the post-engineered DataFrame
        df.to_csv(config['post_processed_csv'], index=False)
        print(f"Saved post-engineered DataFrame to {config['post_processed_csv']}")

    # Now, proceed with training using the post-processed CSV
    feature_columns = [col for col in df.columns if col not in ['isic_id', 'target', 'patient_id']]

    if not feature_columns:
        print("No feature columns identified. Please check the feature engineering step.")
    else:
        print(f"Feature columns identified: {feature_columns}")

    config['lightgbm_config']['feature_columns'] = feature_columns

    print(f"Final feature columns: {config['lightgbm_config']['feature_columns']}")

    if not config['lightgbm_config']['feature_columns']:
        raise ValueError("No feature columns set for LightGBM. Please check the configuration.")

    # Load the post-processed DataFrame for training
    df = pd.read_csv(config['post_processed_csv'])

    print("Starting LightGBM training...")
    lgbm_model_path = train_lightgbm_model(df, config['lightgbm_config'])
    print(f"LightGBM training completed. Models saved at: {lgbm_model_path}")


if __name__ == "__main__":
    main()
