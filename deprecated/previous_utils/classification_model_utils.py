import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
import os
import csv
from tqdm import tqdm
import wandb
from utils.model_head import get_model_head  # Import get_model_head from model_head.py
from utils.classification_evaluation import evaluate_model  # Updated import for evaluation functions

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Combine the modified model and the new head
class ModifiedModel(nn.Module):
    def __init__(self, backbone, head):
        super(ModifiedModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def freeze_subsequential_layers(module, start_idx, end_idx, print_status=False):
    """
    Freeze layers within a Sequential block from start_idx to end_idx.
    """
    for i in range(start_idx, end_idx + 1):
        sub_module = module[i]
        for param in sub_module.parameters():
            param.requires_grad = False
        if print_status:
            print(f"Sub-Sequential {i} within Index 2: Frozen")

def freeze_backbone_layers(backbone, freeze_option, print_status=False):
    """
    Freezes layers of the backbone based on the freeze option.

    Parameters:
    - backbone (nn.Sequential): The backbone model whose layers might be frozen.
    - freeze_option (str): The freeze option. Can be 'unfreeze', 'freezeall', or 'freezecustom1'.
    - print_status (bool): If True, print the status of each layer; if False, do not print.
    """
    if print_status:
        print(f"Freeze option selected: {freeze_option}")

    if freeze_option.lower() == 'freezeall':
        for param in backbone.parameters():
            param.requires_grad = False
        if print_status:
            print("All backbone layers have been frozen.")
    elif freeze_option.lower() == 'unfreeze':
        for param in backbone.parameters():
            param.requires_grad = True
        if print_status:
            print("All backbone layers are unfrozen (trainable).")
    elif freeze_option.lower() == 'freezecustom1':
        # Apply custom freezing logic
        for idx, layer in enumerate(backbone):
            if idx < 2:
                # Freeze all layers at indices 0 and 1
                for param in layer.parameters():
                    param.requires_grad = False
                if print_status:
                    print(f"Index {idx}: Frozen - {layer}")
            elif idx == 2 and isinstance(layer, torch.nn.Sequential):
                # Freeze sub-sequential 0 to 4
                freeze_subsequential_layers(layer, 0, 4, print_status=print_status)
                # Ensure sub-sequential 5 and 6 are unfrozen
                for i in range(5, 7):
                    sub_module = layer[i]
                    for param in sub_module.parameters():
                        param.requires_grad = True
                    if print_status:
                        print(f"Sub-Sequential {i} within Index 2: Unfrozen")
            elif idx >= 3:
                # Do not freeze anything from index 3 onwards
                for param in layer.parameters():
                    param.requires_grad = True
                if print_status:
                    print(f"Index {idx}: Unfrozen - {layer}")
    else:
        raise ValueError(f"Unrecognized freeze option: {freeze_option}. Please choose 'unfreeze', 'freezeall', or 'freezecustom1'.")

def create_model(config, device, pretrained=None):
    """
    Create a model with or without pretrained weights.

    Parameters:
    - config: Configuration dictionary containing model settings.
    - device: Device to run the model on.
    - pretrained: Optional parameter to override the config's 'pretrained' setting.
                  If None, the value from config['pretrained'] will be used.

    Returns:
    - complete_model: The created and configured model.
    """

    # Use the provided 'pretrained' value if specified, otherwise fall back to config.
    if pretrained is None:
        pretrained = config['pretrained']

    # Load the model with or without pretrained weights
    model = timm.create_model(config['model_name'], pretrained=pretrained)

    # Move the model to the device before creating a dummy input
    model = model.to(device)

    # Create a dummy input to pass through the model
    dummy_input = torch.randn(1, 3, config['input_size'][0], config['input_size'][1]).to(device)

    # Modify the backbone depending on the head type
    if config['head_type'].lower() == 'scsa':
        # SCSA head requires spatial information, so we keep the layers that retain spatial dimensions
        modified_backbone = nn.Sequential(*list(model.children())[:-4])
        print("Backbone modified with layers up to -4 (for SCSA head).")
    elif config['head_type'].lower() in ['simplehead', 'simpleheadv2']:
        # SimpleHead and SimpleHeadV2 work with global features, so we remove the classifier layer
        modified_backbone = nn.Sequential(*list(model.children())[:-1])
        print("Backbone modified with layers up to -1 (for SimpleHead or SimpleHeadV2).")
    else:
        raise ValueError(f"Unrecognized head_type: {config['head_type']}. Please choose 'scsa', 'simplehead', or 'simpleheadv2'.")

    # Apply freezing based on the config
    freeze_backbone_layers(modified_backbone, config.get('freeze', 'unfreeze'), config.get('print_backbone', False))

    # Pass the dummy input through the modified backbone to get the output shape
    with torch.no_grad():
        modified_output = modified_backbone(dummy_input)

    # Print the output size of the modified backbone
    print(f"Output size of the modified backbone: {modified_output.shape}")

    # Determine the number of channels in the output of the modified backbone
    input_channels = modified_output.shape[1]

    # Use the modified output as backbone_output_size to initialize SCSAHead
    model_head = get_model_head(config['head_type'], input_channels, modified_output, device)

    # Create the complete modified model
    complete_model = ModifiedModel(modified_backbone, model_head).to(device)  # Ensure all parts of the model are on the device

    return complete_model



def train_model(train_loader, valid_loader, config, device, model_save_prefix):
    # Initialize wandb if logging is enabled
    if config['wandb_log']:
        wandb.init(project=config['project_name'], config=config, reinit=True)
        wandb_config = wandb.config

    # Create the model
    model = create_model(config, device)

    # Choose the loss function
    if config['loss_function'] == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    else:
        criterion = nn.BCELoss()

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Ensure the experiment directory structure exists
    experiment_dir = os.path.join("./logs", model_save_prefix)
    plots_dir = os.path.join(experiment_dir, "plots")
    weights_dir = os.path.join(experiment_dir, "weights")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # Prepare CSV file for saving metrics
    results_csv_path = os.path.join(experiment_dir, "results.csv")
    with open(results_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow(['Epoch', 'Precision', 'Recall', 'F1 Score', 'Log Loss', 'Cohen\'s Kappa', 'Balanced Accuracy', 'MCC'])

    # Training loop with validation every epoch
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["num_epochs"]}', leave=False):
            inputs, labels = inputs.to(device), labels.float().to(device)

            # Ensure batch size is greater than 1 to avoid issues with batch normalization
            if inputs.size(0) <= 1:
                continue

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Loss: {avg_loss:.4f}')

        # Perform validation and evaluate model
        if valid_loader is not None:
            # Evaluate the model (this now includes printing metrics and saving plots if needed)
            metrics = evaluate_model(model, valid_loader, device, epoch=epoch, save_dir=plots_dir)
            precision, recall, f1, log_loss_val, cohens_kappa, balanced_accuracy, mcc = metrics

            # Save metrics to CSV
            with open(results_csv_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([epoch + 1, precision, recall, f1, log_loss_val, cohens_kappa, balanced_accuracy, mcc])

            # Log the loss and other metrics to wandb if logging is enabled
            if config['wandb_log']:
                wandb.log({
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "Log Loss": log_loss_val,
                    "Cohen's Kappa": cohens_kappa,
                    "Balanced Accuracy": balanced_accuracy,
                    "MCC": mcc
                })

        # Save the model weights after each epoch
        model_save_path = os.path.join(weights_dir, f"{model_save_prefix}_epoch{epoch + 1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model weights saved to {model_save_path} after epoch {epoch + 1}")

    # After training is complete, save the final plots
    if valid_loader is not None:
        print("Saving final evaluation plots...")
        evaluate_model(model, valid_loader, device, save_dir=plots_dir)  # Save final evaluation plots

    return model

