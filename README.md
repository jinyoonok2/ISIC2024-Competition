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
### 1. install dependencies
Ensure you have all the necessary Python libraries installed. You can install the required libraries by running:
```
pip install -r requirements.txt
```

### 2. Download the Dataset
Download the dataset from the provided Google Drive link and place it in the appropriate folder as specified in the configuration file.

### 3. Train Vision Models
To train the vision models using the timm library, you can either run the Python script or use the Jupyter notebook:

- option 1: Run the Python script vision_train.py. This will train the models as defined in the config dictionary, where you can specify which models (e.g., EfficientNet, SelecSLS) you want to train from the timm library.
```
python vision_train.py
```
- option 2: Use the Jupyter notebook train_vast_vision.ipynb to train the models interactively online. You can modify the config variable inside the notebook to select the vision models you'd like to train.

After the training is complete, the features extracted by these models will be saved and can be used for further processing.

### 4. Generate Metadata with Extracted Features
Once the vision models are trained, the extracted features from these models will be added to the metadata. Run evaluation_cv.py to generate the final metadata CSV file. This file will include:

- Feature engineering outputs.
- Processed columns with missing values handled.
- Predictions from trained vision models added as new features to the metadata.

```
python evaluation_cv.py
```

### 5. Train LightGBM and CatBoost Models
The newly generated metadata (which includes both image-based and tabular features) is now used to train LightGBM and CatBoost models. These models are trained using cross-validation to ensure robustness. To train the models, run the following script:

```
python main_cv.py
```
This script will:
- Use cross-validation to train multiple models.
- Aggregate results across folds to provide an average performance metric.

### 6. Submit Results
Once all models have been trained, you can generate the final submission file using the notebook isic2024-ensemble-learning-cv.ipynb. This notebook combines the predictions from different models and prepares the results for submission.


