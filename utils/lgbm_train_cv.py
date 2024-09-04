import lightgbm as lgb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
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
    group_column = config['group_column']
    n_splits = config['n_splits']
    seed = config['seed']
    display_feature_importance = config['display_feature_importance']
    log_dir = config['log_dir']

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

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    lgb_models = []
    oof_df = pd.DataFrame()
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[target_column], groups=df[group_column])):
        print(f"Training fold {fold + 1}/{n_splits}...")

        # Create directories for each fold
        fold_model_dir = os.path.join(log_dir, f"fold_{fold + 1}")
        os.makedirs(fold_model_dir, exist_ok=True)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[val_idx].reset_index(drop=True)

        # Debug: Print shapes and column names to ensure they are correct
        print(f"Fold {fold + 1} Train shape: {train_df.shape}, Validation shape: {valid_df.shape}")
        print(f"Feature columns: {feature_columns}")

        if len(feature_columns) == 0:
            raise ValueError("Feature columns are empty. Please check your configuration.")

        dtrain = lgb.Dataset(train_df[feature_columns], label=train_df[target_column])
        dvalid = lgb.Dataset(valid_df[feature_columns], label=valid_df[target_column])

        # Create early stopping callback
        early_stopping_callback = lgb.early_stopping(
            stopping_rounds=50,  # Number of rounds without improvement before stopping
            first_metric_only=False,  # Whether to use only the first metric for early stopping
            verbose=True,  # Whether to log early stopping messages
            min_delta=0.001  # Minimum improvement in score to keep training
        )

        # Train the model with the callback
        model = lgb.train(
            lgbm_config,
            dtrain,
            valid_sets=[dtrain, dvalid],
            callbacks=[early_stopping_callback]
        )

        # Save the model
        model_save_path = os.path.join(fold_model_dir, "best_model.txt")
        model.save_model(model_save_path)
        print(f"Best model for fold {fold + 1} saved to {model_save_path}")

        valid_df['pred'] = model.predict(valid_df[feature_columns])
        fold_score = comp_score(valid_df[target_column], valid_df['pred'])
        fold_scores.append(fold_score)
        print(f"Fold {fold + 1} - Partial AUC Score: {fold_score:.5f}")

        oof_df = pd.concat([oof_df, valid_df[['isic_id', target_column, 'pred']]])

        lgb_models.append(model)

    lgbm_score = comp_score(oof_df['target'], oof_df['pred'])
    print(f"\nOverall LGBM Score (Partial AUC): {lgbm_score:.5f}")

    if display_feature_importance:
        importances = np.mean([model.feature_importance(importance_type='gain') for model in lgb_models], axis=0)
        df_imp = pd.DataFrame({"feature": feature_columns, "importance": importances}).sort_values(
            "importance", ascending=False).reset_index(drop=True)

        plt.figure(figsize=(16, 12))
        plt.barh(df_imp["feature"], df_imp["importance"])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.show()

    print(f"Training completed. Models saved under: {log_dir}")
    return log_dir