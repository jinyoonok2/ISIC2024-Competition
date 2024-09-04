import torch
import timm
from utils.model_utils import create_model, freeze_backbone_layers, freeze_subsequential_layers

# Configuration dictionary for the model
CONFIG = {
    'model_name': 'efficientnet_b3.ra2_in1k',  # You can replace this with the model you want to use
    'pretrained': True,
    'input_size': (128, 128),  # Input size for the model
    'head_type': 'simplehead',  # Head type ('simplehead' or 'scsa')
    'freeze': 'freezecustom1'  # Custom freeze option
}

def create_model(config, device):
    # Load the pretrained model
    model = timm.create_model(config['model_name'], pretrained=config['pretrained'])

    # Move the model to the device
    model = model.to(device)

    # Modify the backbone depending on the head type
    if config['head_type'].lower() == 'scsa':
        modified_backbone = torch.nn.Sequential(*list(model.children())[:-4])
    elif config['head_type'].lower() in ['simplehead', 'simpleheadv2']:
        modified_backbone = torch.nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError(f"Unrecognized head_type: {config['head_type']}")

    # Apply custom freezing based on Sequential indices
    freeze_backbone_layers(modified_backbone, config.get('freeze', 'unfreeze'), print_status=True)

    return modified_backbone

def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model using the create_model function
    model = create_model(CONFIG, device)

    # Print the modified backbone structure with indices and freeze status
    print("\nModified Backbone Structure with Indices and Freeze Status:\n")
    for idx, layer in enumerate(model):
        if idx == 2 and isinstance(layer, torch.nn.Sequential):
            for sub_idx, sub_layer in enumerate(layer):
                frozen = all(not param.requires_grad for param in sub_layer.parameters())
                status = "Frozen" if frozen else "Unfrozen"
                print(f"Index 2 - Sub-Sequential {sub_idx}: {status} - {sub_layer}")
        else:
            frozen = all(not param.requires_grad for param in layer.parameters())
            status = "Frozen" if frozen else "Unfrozen"
            print(f"Index {idx}: {status} - {layer}")

if __name__ == "__main__":
    main()