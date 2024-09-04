import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def comp_score(solution: pd.DataFrame, submission: pd.DataFrame, min_tpr: float = 0.80):
    v_gt = abs(np.asarray(solution) - 1)
    v_pred = np.array([1.0 - x for x in submission])
    max_fpr = abs(1 - min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc

def train_lightgbm_model(df, config):
    lgbm_config = config['lgb_params']
    feature_columns = config['feature_columns']
    target_column = config['target_column']
    log_dir = config['log_dir']
    display_feature_importance = config['display_feature_importance']

    # Ensure the logging directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Print the feature columns being used
    print("Feature columns being used:")
    print(feature_columns)
    print(f"Number of feature columns: {len(feature_columns)}")

    # Data validation: Ensure all feature columns are numeric
    non_numeric_cols = df[feature_columns].select_dtypes(exclude=[np.number]).columns
    if not non_numeric_cols.empty:
        print(f"Non-numeric data found in columns: {non_numeric_cols.tolist()}")

    # Check for missing values
    missing_values = df[feature_columns].isnull().sum()
    if missing_values.any():
        print(f"Missing values found in the following columns:")
        print(missing_values[missing_values > 0])

        # Print the location of missing values
        print("Location of missing values:")
        print(df[feature_columns].isnull().stack()[lambda x: x].index.tolist())

    # Proceed only if there are no non-numeric columns and no missing values
    assert all(df[feature_columns].dtypes.apply(lambda x: np.issubdtype(x, np.number))), "Non-numeric data found in features"
    assert not df[feature_columns].isnull().values.any(), "Missing values found in features"

    # Train the model on the full dataset
    dtrain = lgb.Dataset(df[feature_columns], label=df[target_column])

    # Train the model without early stopping
    model = lgb.train(
        lgbm_config,
        dtrain
    )

    # Save the model
    model_save_path = os.path.join(log_dir, "best_model.txt")
    model.save_model(model_save_path)
    print(f"Best model saved to {model_save_path}")

    if display_feature_importance:
        importances = model.feature_importance(importance_type='gain')
        df_imp = pd.DataFrame({"feature": feature_columns, "importance": importances}).sort_values(
            "importance", ascending=False).reset_index(drop=True)

        plt.figure(figsize=(16, 12))
        plt.barh(df_imp["feature"], df_imp["importance"])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.show()

    print(f"Training completed. Model saved under: {model_save_path}")
    return model_save_path

