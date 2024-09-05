# ISIC 2024 Competition - Skin Lesion Classification

This repository contains implementations for the ISIC 2024 Competition. It includes code for training vision models using the timm library and integrating them with tabular feature engineering, alongside classical machine learning models like LightGBM and CatBoost for skin lesion classification.

## Table of Contents
- [Overview](#overview)
- [Model Weights](#model-weights)
- [Dataset](#dataset)
- [Models and Features](#models-and-features)
- [Pipeline Structure](#pipeline-structure)
- [Running the Code](#running-the-code)
- [Feature Engineering](#feature-engineering)
- [LightGBM and CatBoost Training](#lightgbm-and-catboost-training)
- [References](#references)

# Overview

This repository is designed to train both deep learning models and machine learning models for the ISIC 2024 skin lesion classification challenge. The key components include:

1. Vision Models using the timm library to extract features from images.
2. Tabular Feature Engineering for metadata and additional attributes.
3. Classical Machine Learning Models such as LightGBM and CatBoost that integrate image features with engineered features to make final predictions.

# Model Weights
The model weights used in this project are saved and available for download via the following Google Drive link:
https://drive.google.com/drive/folders/10qHWaomKA4xAByLHZ-SYHsD72n_oM16n?usp=drive_link


# Dataset
The dataset used for this competition, including the metadata and image files, can be accessed through this link:
https://drive.google.com/drive/folders/16fvgCYDsqvByeaw2J6Vlv-8SQ1AIyMlR

# Models and Features
### Vision Models:
- EfficientNet-B3 (efficientnet_b3.ra2_in1k): A high-performance, efficient convolutional neural network.
- SelecSLS42 (selecsls42b.in1k): A fast and lightweight architecture.
- NextViT (nextvit_small.bd_in1k_384): A transformer-based vision model with enhanced feature extraction.

These models are used to extract deep features from the images, which are then used in combination with tabular features in the downstream classification models.

```
# Example of loading a pretrained model and generating features
model_name = 'efficientnet_b3.ra2_in1k'
model = timm.create_model(model_name, pretrained=True, num_classes=1)
model.load_state_dict(torch.load(config['model_paths'][model_name]))

# Pass images through the model and extract features
features = model(images)
```

### Tabular Features:
The tabular features include metadata and engineered features such as:
- Lesion size ratio: Ratio of minor axis to clinical long diameter.
- Color contrast: Differences in lesion colors using the tbp_lv features.
- 3D lesion orientation: Calculated based on the tbp_lv_x, tbp_lv_y, and tbp_lv_z values.

```
# Example of feature engineering
df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
df["3d_position_distance"] = np.sqrt(df["tbp_lv_x"]**2 + df["tbp_lv_y"]**2 + df["tbp_lv_z"]**2)
```

# Pipeline Structure
1. Data Preprocessing:
  - Load the dataset and perform data cleaning (e.g., handle missing values, drop unnecessary columns).
  - Apply feature engineering on tabular data to create new columns that capture essential lesion characteristics.
2. Deep Learning Feature Extraction:
  - Use pretrained models to extract deep features from the images.
  - Save the extracted features to the dataset as additional columns for further processing.
3. Training with LightGBM and CatBoost:
  - Combine the image features with the tabular features.
  - Train LightGBM and CatBoost models using cross-validation.
4. Post-processing:
  - After training, the models are evaluated, and the predictions are saved for submission.

# Running the Code
Ensure you have all the necessary Python libraries installed. You can install the required libraries by running:
```
pip install -r requirements.txt
```
Download the dataset from the provided Google Drive link and place it in the appropriate folder as specified in the configuration file.

