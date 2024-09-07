import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
from tqdm import tqdm
import torch.nn as nn
import h5py
from io import BytesIO

# Configuration dictionary
config = {
    'batch_size': 32,
    'model_path': './logs/logs_effb3ra2in1k/epoch_19.pth',  # EfficientNet model only
    'lightgbm_model_paths': [
        './logs/lightgbm/fold_1/best_model.txt',
        './logs/lightgbm/fold_2/best_model.txt',
        './logs/lightgbm/fold_3/best_model.txt',
        './logs/lightgbm/fold_4/best_model.txt',
        './logs/lightgbm/fold_5/best_model.txt'
    ],
    'submission_csv': './datasets/isic-2024-challenge/submission.csv',
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

# Non-numeric data found in columns
cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple", "anatom_site_general"]

# Columns to drop based on comparison with test-metadata
columns_to_drop = [
    'iddx_3', 'iddx_2', 'iddx_5', 'iddx_full',
    'tbp_lv_dnn_lesion_confidence', 'les1ion_id',
    'mel_mitotic_index', 'mel_thick_mm', 'iddx_1', 'iddx_4',
    'image_type', 'attribution', 'copyright_license'
]

# Custom Dataset Class for HDF5 Images
class CustomDataset(Dataset):
    def __init__(self, hdf5_file, isic_ids, transform=None):
        self.hdf5_file = hdf5_file
        self.isic_ids = isic_ids
        self.transform = transform
        self.fp_hdf = h5py.File(hdf5_file, mode="r")

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx):
        isic_id = self.isic_ids[idx]
        img = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img

# DataLoader Function with Model-Specific Transforms
def get_dataloader(hdf5_file, isic_ids, model_name, batch_size):
    base_model = timm.create_model(model_name, pretrained=False, num_classes=1)
    data_config = timm.data.resolve_model_data_config(base_model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    dataset = CustomDataset(hdf5_file=hdf5_file, isic_ids=isic_ids, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

# Function to generate features using the EfficientNet model
def generate_features(model_name, model_path, hdf5_file, isic_ids, df, config):
    print(f"Generating features using model: {model_name}")

    base_model = timm.create_model(model_name, pretrained=False, num_classes=1)
    model = nn.Sequential(base_model, nn.Sigmoid())  # Sigmoid layer to output probabilities
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = get_dataloader(hdf5_file, isic_ids, model_name, config['batch_size'])
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

# Function to load LightGBM models
def load_models(model_paths):
    import lightgbm as lgb
    models = [lgb.Booster(model_file=path) for path in model_paths]
    return models

# Function to predict test data
def predict_test_data():
    df_test = pd.read_csv('./datasets/isic-2024-challenge/test-metadata.csv')
    original_isic_ids = df_test['isic_id'].copy()

    # Drop unnecessary columns
    df_test = df_test.drop(columns=columns_to_drop, errors='ignore')

    # Fill missing values with median in numeric columns
    df_test = fill_na_with_median(df_test, num_cols)

    hdf5_file = './datasets/isic-2024-challenge/test-image.hdf5'
    isic_ids = df_test['isic_id'].tolist()

    # Generate features using EfficientNet model
    df_test = generate_features('efficientnet_b3.ra2_in1k', config['model_path'], hdf5_file, isic_ids, df_test, config)

    # Encode categorical columns
    df_test, _ = encode_categorical(df_test, cat_cols)

    # Fill remaining missing values with -1
    df_test.fillna(-1, inplace=True)

    # Ensure the column order matches the order used in the model training
    feature_columns = [col for col in df_test.columns if col not in ['isic_id', 'patient_id']]
    df_test = df_test[feature_columns]

    # Load trained LightGBM models
    lgb_models = load_models(config['lightgbm_model_paths'])

    # Make predictions with LightGBM models
    lgb_preds = np.mean([model.predict(df_test) for model in lgb_models], axis=0)

    # Create submission DataFrame
    submission = pd.DataFrame({
        'isic_id': original_isic_ids,  # Use the original isic_id from test-metadata.csv
        'target': lgb_preds
    })

    # Print the first 5 rows of the submission DataFrame
    print("First 5 rows of the submission DataFrame:")
    print(submission.head())

    # Save submission to CSV
    submission.to_csv(config['submission_csv'], index=False)
    print(f"Submission saved to {config['submission_csv']}")

if __name__ == "__main__":
    predict_test_data()
