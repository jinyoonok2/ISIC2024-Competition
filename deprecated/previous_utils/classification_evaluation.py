import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve,
    log_loss, cohen_kappa_score, balanced_accuracy_score,
    matthews_corrcoef
)
import os
import torchvision.utils as vutils
import torch
from tqdm import tqdm


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot and display or save a normalized confusion matrix.

    Inputs:
    - y_true: Actual labels (ground truth).
    - y_pred: Predicted labels by the model.
    - class_names: List of class names for labeling the matrix.
    - save_path: If provided, save the plot to this path.

    Output:
    - A heatmap visualization showing the normalized counts of true positives (TP), true negatives (TN),
      false positives (FP), and false negatives (FN).

    Interpretation:
    - Helps in understanding how many instances of each class are correctly or incorrectly classified.
      Normalization ensures that smaller population classes can be effectively visualized.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')  # Normalize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)

    plt.title("Normalized Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def print_classification_report(y_true, y_pred):
    """
    Print precision, recall, and F1 score.

    Inputs:
    - y_true: Actual labels (ground truth).
    - y_pred: Predicted labels by the model.

    Outputs:
    - Precision: The ratio of correctly predicted positive observations to total predicted positives.
    - Recall: The ratio of correctly predicted positive observations to all observations in the actual positive class.
    - F1 Score: The harmonic mean of precision and recall.

    Interpretation:
    - Precision: High precision indicates that the model produces few false positives.
    - Recall: High recall indicates that the model captures most of the actual positives.
    - F1 Score: Balances precision and recall; useful when the class distribution is imbalanced.
    """
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


def plot_roc_curve(y_true, y_pred_prob, save_path=None):
    """
    Plot and display or save an ROC curve with a horizontal line at TPR = 0.8.

    Inputs:
    - y_true: Actual labels (ground truth).
    - y_pred_prob: Predicted probabilities for the positive class.
    - save_path: If provided, save the plot to this path.

    Outputs:
    - A plot showing the ROC curve, which plots the true positive rate (recall) against the false positive rate.
    - AUC (Area Under the Curve) is displayed on the plot, which represents the overall ability of the model to distinguish between positive and negative classes.

    Interpretation:
    - AUC close to 1: Model has excellent performance.
    - AUC close to 0.5: Model performance is no better than random guessing.
    - The horizontal line at TPR = 0.8 helps to emphasize the model's performance threshold.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')

    # Add a horizontal line at TPR = 0.8
    plt.axhline(y=0.8, color='red', linestyle='--', label='TPR = 0.8')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_prob, save_path=None):
    """
    Plot and display or save a Precision-Recall curve.

    Inputs:
    - y_true: Actual labels (ground truth).
    - y_pred_prob: Predicted probabilities for the positive class.
    - save_path: If provided, save the plot to this path.

    Outputs:
    - A plot showing the precision-recall curve, which plots precision against recall at different thresholds.

    Interpretation:
    - High precision and high recall are desirable.
    - The curve helps in understanding the trade-off between precision and recall at various thresholds.
    - Particularly useful when dealing with imbalanced datasets.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def compute_log_loss(y_true, y_pred_prob):
    """
    Compute and print Logarithmic Loss (Log Loss).

    Inputs:
    - y_true: Actual labels (ground truth).
    - y_pred_prob: Predicted probabilities for the positive class.

    Output:
    - Log Loss: A scalar value that measures the accuracy of probabilistic predictions.

    Interpretation:
    - Lower Log Loss indicates better performance.
    - Log Loss penalizes false confident predictions heavily, making it useful to monitor during training.
    """
    loss = log_loss(y_true, y_pred_prob)
    print(f"Log Loss: {loss:.4f}")
    return loss

def compute_cohens_kappa(y_true, y_pred):
    """
    Compute and print Cohen's Kappa.

    Inputs:
    - y_true: Actual labels (ground truth).
    - y_pred: Predicted labels by the model.

    Output:
    - Cohen's Kappa: A scalar value ranging from -1 to 1 that measures agreement between the true labels and predicted labels, adjusting for chance.

    Interpretation:
    - Kappa close to 1: Strong agreement.
    - Kappa close to 0: Agreement is no better than chance.
    - Kappa close to -1: Strong disagreement.
    """
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa:.4f}")
    return kappa

def compute_balanced_accuracy(y_true, y_pred):
    """
    Compute and print Balanced Accuracy.

    Inputs:
    - y_true: Actual labels (ground truth).
    - y_pred: Predicted labels by the model.

    Output:
    - Balanced Accuracy: A scalar value that is the average of recall obtained on each class. It considers class imbalance.

    Interpretation:
    - Higher balanced accuracy indicates better performance, especially on imbalanced datasets.
    """
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    return balanced_acc

def compute_mcc(y_true, y_pred):
    """
    Compute and print Matthews Correlation Coefficient (MCC).

    Inputs:
    - y_true: Actual labels (ground truth).
    - y_pred: Predicted labels by the model.

    Output:
    - MCC: A scalar value ranging from -1 to 1 that measures the quality of binary classifications, considering TP, TN, FP, and FN.

    Interpretation:
    - MCC close to 1: Perfect prediction.
    - MCC close to 0: No better than random prediction.
    - MCC close to -1: Total disagreement between prediction and observation.
    """
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    return mcc

def evaluate_model(model, valid_loader, device, epoch=None, save_dir=None):
    """
    Evaluate the model using various metrics and save the results/plots if a save directory is provided.

    If `epoch` is `None`, plots will be saved regardless of epoch number.
    """

    model.eval()  # Set the model to evaluation mode
    all_targets = []
    all_predictions = []
    all_predictions_prob = []

    with torch.no_grad():
        for inputs, targets in tqdm(valid_loader, desc=f"Evaluating Epoch {epoch}"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            probs = outputs.cpu().numpy()
            predictions = (probs > 0.5).astype(int)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions)
            all_predictions_prob.extend(probs)

    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_predictions_prob = np.array(all_predictions_prob)

    # Compute metrics
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    log_loss_val = compute_log_loss(all_targets, all_predictions_prob)
    cohens_kappa = compute_cohens_kappa(all_targets, all_predictions)
    balanced_accuracy = compute_balanced_accuracy(all_targets, all_predictions)
    mcc = compute_mcc(all_targets, all_predictions)

    # Print metrics
    print(f"Epoch {epoch + 1 if epoch is not None else 'N/A'} Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Log Loss: {log_loss_val:.4f}")
    print(f"Cohen's Kappa: {cohens_kappa:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"MCC: {mcc:.4f}")

    # Save plots every 5 epochs or always if epoch is None
    if save_dir and (epoch is None or (epoch + 1) % 5 == 0):
        os.makedirs(save_dir, exist_ok=True)
        epoch_str = f"_epoch{epoch + 1}" if epoch is not None else ""
        plot_confusion_matrix(all_targets, all_predictions, ['Class 0', 'Class 1'], save_path=os.path.join(save_dir, f'confusion_matrix{epoch_str}.png'))
        plot_roc_curve(all_targets, all_predictions_prob, save_path=os.path.join(save_dir, f'roc_curve{epoch_str}.png'))
        plot_precision_recall_curve(all_targets, all_predictions_prob, save_path=os.path.join(save_dir, f'precision_recall_curve{epoch_str}.png'))

    return precision, recall, f1, log_loss_val, cohens_kappa, balanced_accuracy, mcc



def visualize_dataloader_batch(dataloader, title="Batch from DataLoader"):
    """
    Visualize a batch of images from a DataLoader.

    Inputs:
    - dataloader: A PyTorch DataLoader providing batches of images and labels.
    - title: Title of the plot.

    Output:
    - Displays a grid of images from the DataLoader batch.
    """
    # Get a batch of images and labels
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # Create a grid of images
    img_grid = vutils.make_grid(images, normalize=True, scale_each=True)

    # Plot the grid
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(img_grid.permute(1, 2, 0))  # Change from CHW to HWC for plotting
    plt.axis('off')
    plt.show()