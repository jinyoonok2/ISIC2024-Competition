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
    'model_paths': {
        'selecsls42b.in1k': './logs/logs_selecsls42bin1k/epoch_35.pth',
        'efficientnet_b0.ra_in1k': "./logs/logs_effb3ra2in1k/epoch_19.pth",
        'nextvit_small.bd_in1k_384' : "./logs/logs_nextvit_small/nextvit_small.bd_in1k_384/epoch_9.pth",
    },
    'lightgbm_model_paths': ['./logs/lightgbm/fold_1/best_model.txt',
                             './logs/lightgbm/fold_2/best_model.txt',
                             './logs/lightgbm/fold_3/best_model.txt',
                             './logs/lightgbm/fold_4/best_model.txt',
                             './logs/lightgbm/fold_5/best_model.txt'],
    'catboost_model_paths': ['./logs/catboost/fold_1/best_model.cbm',
                             './logs/catboost/fold_2/best_model.cbm',
                             './logs/catboost/fold_3/best_model.cbm',
                             './logs/catboost/fold_4/best_model.cbm',
                             './logs/catboost/fold_5/best_model.cbm'],
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

# Non-numeric data found in columns: ['anatom_site_general', 'image_type', 'attribution', 'copyright_license', 'combined_anatomical_site']
cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple", "anatom_site_general","combined_anatomical_site", ]

# List of columns to drop based on comparison with test-metadata
columns_to_drop = [
    'iddx_3', 'iddx_2', 'iddx_5', 'iddx_full',
    'tbp_lv_dnn_lesion_confidence', 'lesion_id',
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
        # Corrected image loading using BytesIO
        img = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))

        if self.transform:
            img = Image.fromarray(img)  # Convert NumPy array to PIL Image
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

# Function to generate features using the saved model
def generate_features(model_name, model_path, hdf5_file, isic_ids, df, config):
    print(f"Generating features using model: {model_name}")

    base_model = timm.create_model(model_name, pretrained=False, num_classes=1)

    model = nn.Sequential(
        base_model,
        nn.Sigmoid()  # Sigmoid layer to output probabilities
    )

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

# Function to load models
def load_models(model_paths, model_type):
    models = []
    if model_type == 'lightgbm':
        import lightgbm as lgb
        for path in model_paths:
            model = lgb.Booster(model_file=path)
            models.append(model)
    elif model_type == 'catboost':
        from catboost import CatBoostClassifier
        for path in model_paths:
            model = CatBoostClassifier()
            model.load_model(path)
            models.append(model)
    return models

def compare_train_test_columns(train_df, test_df):
    train_columns = set(train_df.columns)
    test_columns = set(test_df.columns)

    print("Columns in the training data but not in the test data:")
    print(train_columns - test_columns)

    print("\nColumns in the test data but not in the training data:")
    print(test_columns - train_columns)

    print("\nTraining Data Columns:")
    print(sorted(train_columns))

    print("\nTest Data Columns:")
    print(sorted(test_columns))

def predict_test_data():
    df_test = pd.read_csv('./datasets/isic-2024-challenge/test-metadata.csv')

    # Retain original isic_id for submission
    original_isic_ids = df_test['isic_id'].copy()

    # Drop unnecessary columns
    df_test = df_test.drop(columns=columns_to_drop, errors='ignore')

    # Fill missing values with median in numeric columns
    df_test = fill_na_with_median(df_test, num_cols)

    # Apply feature engineering
    df_test = feature_engineering(df_test)

    hdf5_file = './datasets/isic-2024-challenge/test-image.hdf5'
    isic_ids = df_test['isic_id'].tolist()

    # Load and apply feature generation models
    for model_name, model_path in config['model_paths'].items():
        df_test = generate_features(model_name, model_path, hdf5_file, isic_ids, df_test, config)

    # Encode categorical columns
    df_test, _ = encode_categorical(df_test, cat_cols)

    # Fill remaining missing values with -1
    df_test.fillna(-1, inplace=True)

    # Ensure the column order matches the order that was used in the model training
    feature_columns = [col for col in df_test.columns if col not in ['isic_id', 'patient_id']]
    df_test = df_test[feature_columns]

    # Load trained models
    lgb_models = load_models(config['lightgbm_model_paths'], 'lightgbm')
    cb_models = load_models(config['catboost_model_paths'], 'catboost')

    # Make predictions with LightGBM models
    lgb_preds = np.mean([model.predict(df_test) for model in lgb_models], axis=0)

    # Make predictions with CatBoost models
    cb_preds = np.mean([model.predict_proba(df_test)[:, 1] for model in cb_models], axis=0)

    # Ensemble the predictions
    preds = lgb_preds * 0.315 + cb_preds * 0.685

    # Create submission DataFrame
    submission = pd.DataFrame({
        'isic_id': original_isic_ids,  # Use the original isic_id from test-metadata.csv
        'target': preds
    })

    # Print the first 5 rows of the submission DataFrame
    print("First 5 rows of the submission DataFrame:")
    print(submission.head())

    # Save submission to CSV
    submission.to_csv(config['submission_csv'], index=False)
    print(f"Submission saved to {config['submission_csv']}")



# Feature Engineering Function
def feature_engineering(df):
    df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
    df["lesion_shape_index"] = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
    df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
    df["luminance_contrast"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
    df["lesion_color_difference"] = np.sqrt(
        df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2)
    df["border_complexity"] = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
    df["color_uniformity"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]
    df["3d_position_distance"] = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2)
    df["perimeter_to_area_ratio"] = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
    df["area_to_perimeter_ratio"] = df["tbp_lv_areaMM2"] / df["tbp_lv_perimeterMM"]
    df["lesion_visibility_score"] = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]
    df["combined_anatomical_site"] = df["anatom_site_general"] + "_" + df["tbp_lv_location"]
    df["symmetry_border_consistency"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
    df["consistency_symmetry_border"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"] / (
            df["tbp_lv_symm_2axis"] + df["tbp_lv_norm_border"])
    df["color_consistency"] = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]
    df["consistency_color"] = df["tbp_lv_stdL"] * df["tbp_lv_Lext"] / (df["tbp_lv_stdL"] + df["tbp_lv_Lext"])
    df["size_age_interaction"] = df["clin_size_long_diam_mm"] * df["age_approx"]
    df["hue_color_std_interaction"] = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
    df["lesion_severity_index"] = (df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]) / 3
    df["shape_complexity_index"] = df["border_complexity"] + df["lesion_shape_index"]
    df["color_contrast_index"] = df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df[
        "tbp_lv_deltaLBnorm"]
    df["log_lesion_area"] = np.log(df["tbp_lv_areaMM2"] + 1)
    df["normalized_lesion_size"] = df["clin_size_long_diam_mm"] / df["age_approx"]
    df["mean_hue_difference"] = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
    df["std_dev_contrast"] = np.sqrt(
        (df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3)
    df["color_shape_composite_index"] = (df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df[
        "tbp_lv_symm_2axis"]) / 3
    df["3d_lesion_orientation"] = np.arctan2(df["tbp_lv_y"], df["tbp_lv_x"])
    df["overall_color_difference"] = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3
    df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
    df["comprehensive_lesion_index"] = (df["tbp_lv_area_perim_ratio"] + df["tbp_lv_eccentricity"] + df[
        "tbp_lv_norm_color"] + df["tbp_lv_symm_2axis"]) / 4
    df["color_variance_ratio"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_stdLExt"]
    df["border_color_interaction"] = df["tbp_lv_norm_border"] * df["tbp_lv_norm_color"]
    df["size_color_contrast_ratio"] = df["clin_size_long_diam_mm"] / df["tbp_lv_deltaLBnorm"]
    df["age_normalized_nevi_confidence"] = df["tbp_lv_nevi_confidence"] / df["age_approx"]
    df["color_asymmetry_index"] = df["tbp_lv_radial_color_std_max"] * df["tbp_lv_symm_2axis"]
    df["3d_volume_approximation"] = df["tbp_lv_areaMM2"] * np.sqrt(
        df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2)
    df["color_range"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs() + (df["tbp_lv_A"] - df["tbp_lv_Aext"]).abs() + (
            df["tbp_lv_B"] - df["tbp_lv_Bext"]).abs()
    df["shape_color_consistency"] = df["tbp_lv_eccentricity"] * df["tbp_lv_color_std_mean"]
    df["border_length_ratio"] = df["tbp_lv_perimeterMM"] / (2 * np.pi * np.sqrt(df["tbp_lv_areaMM2"] / np.pi))
    df["age_size_symmetry_index"] = df["age_approx"] * df["clin_size_long_diam_mm"] * df["tbp_lv_symm_2axis"]
    df["index_age_size_symmetry"] = df["age_approx"] * df["tbp_lv_areaMM2"] * df["tbp_lv_symm_2axis"]

    return df


if __name__ == "__main__":
    predict_test_data()
