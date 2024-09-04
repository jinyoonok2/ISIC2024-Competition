import os
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import logging

def comp_score(solution: pd.DataFrame, submission: pd.DataFrame, min_tpr: float = 0.80):
    v_gt = abs(np.asarray(solution) - 1)
    v_pred = np.array([1.0 - x for x in submission])
    max_fpr = abs(1 - min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc

def train_catboost_model(df, config):
    cb_params = config['cb_params']
    feature_columns = config['feature_columns']
    target_column = config['target_column']
    group_column = config.get('group_column', None)
    n_splits = config['n_splits']
    seed = config['seed']
    cat_features = config['cat_features']
    display_feature_importance = config['display_feature_importance']
    log_dir = config['log_dir']

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Ensure the logging directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Data validation: Ensure numeric features are numeric and handle categorical features appropriately
    if cat_features:
        numeric_features = [col for col in feature_columns if col not in cat_features]
        assert all(df[numeric_features].dtypes.apply(lambda x: np.issubdtype(x, np.number))), "Non-numeric data found in numeric features"
    else:
        numeric_features = feature_columns
        assert all(df[numeric_features].dtypes.apply(lambda x: np.issubdtype(x, np.number))), "Non-numeric data found in features"

    assert not df[feature_columns].isnull().values.any(), "Missing values found in features"

    # Initialize cross-validator
    if group_column:
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    cb_models = []
    oof_df = pd.DataFrame()
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[target_column], groups=df[group_column] if group_column else None)):
        logging.info(f"Training fold {fold + 1}/{n_splits}...")

        # Create directories for each fold
        fold_model_dir = os.path.join(log_dir, f"fold_{fold + 1}")
        os.makedirs(fold_model_dir, exist_ok=True)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[val_idx].reset_index(drop=True)

        model = cb.CatBoostClassifier(**cb_params)
        model.fit(
            train_df[feature_columns],
            train_df[target_column],
            eval_set=(valid_df[feature_columns], valid_df[target_column]),
            cat_features=cat_features,
            verbose=100,
            use_best_model=True,
            save_snapshot=False  # Disable snapshotting
        )

        # Save the model at the best iteration
        model_save_path = os.path.join(fold_model_dir, "best_model.cbm")
        model.save_model(model_save_path)
        logging.info(f"Best model for fold {fold + 1} saved to {model_save_path}")

        # Predict on validation set
        valid_df['pred'] = model.predict_proba(valid_df[feature_columns])[:, 1]
        fold_score = comp_score(valid_df[target_column], valid_df['pred'])
        fold_scores.append(fold_score)
        logging.info(f"Fold {fold + 1} - Partial AUC Score: {fold_score:.5f}")

        # Collect out-of-fold predictions
        if 'isic_id' in df.columns:
            oof_df = pd.concat([oof_df, valid_df[['isic_id', target_column, 'pred']]], ignore_index=True)
        else:
            oof_df = pd.concat([oof_df, valid_df[[target_column, 'pred']]], ignore_index=True)

        cb_models.append(model)

    cb_score = np.mean(fold_scores)
    logging.info(f"\nOverall CatBoost Score (Partial AUC): {cb_score:.5f}")

    if display_feature_importance:
        # Calculate median feature importance across folds
        importances = np.median([model.get_feature_importance() for model in cb_models], axis=0)
        df_imp = pd.DataFrame({"feature": feature_columns, "importance": importances}).sort_values(
            "importance", ascending=False).reset_index(drop=True)

        plt.figure(figsize=(16, 12))
        plt.barh(df_imp["feature"], df_imp["importance"])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()  # Highest importance at the top
        plt.show()

    logging.info(f"Training completed. Models saved under: {log_dir}")
    return log_dir
