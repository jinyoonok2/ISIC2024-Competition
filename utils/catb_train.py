import os
import pandas as pd
import numpy as np
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


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
    cat_features = config['cat_features']
    display_feature_importance = config['display_feature_importance']
    log_dir = config['log_dir']

    # Ensure the logging directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Data validation: Ensure all feature columns are numeric and have no missing values
    assert all(df[feature_columns].dtypes.apply(lambda x: np.issubdtype(x, np.number))), "Non-numeric data found in features"
    assert not df[feature_columns].isnull().values.any(), "Missing values found in features"

    # Train the model on the full dataset
    model = cb.CatBoostClassifier(**cb_params)
    model.fit(df[feature_columns], df[target_column],
              cat_features=cat_features,
              verbose=100,
              save_snapshot=True,
              snapshot_file=os.path.join(log_dir, "snapshot.cbsnapshot"))

    # Save the model
    model_save_path = os.path.join(log_dir, "best_model.cbm")
    model.save_model(model_save_path)
    print(f"Best model saved to {model_save_path}")

    if display_feature_importance:
        importances = model.get_feature_importance()
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
