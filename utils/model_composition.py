import timm
import torch

def print_model_structure(model_name):
    print(f"\nModel: {model_name}")
    model = timm.create_model(model_name, pretrained=True, num_classes=1)  # Binary classification
    print(model)
    print("="*80)

def print_model_data_config(model_name):
    print(f"\nData Config for Model: {model_name}")
    model = timm.create_model(model_name, pretrained=True, num_classes=1)
    data_config = timm.data.resolve_model_data_config(model)
    for key, value in data_config.items():
        print(f"{key}: {value}")
    print("="*80)

def main():
    model_names = [
        'selecsls42b.in1k',
        'nextvit_small.bd_in1k_384',
        'efficientnet_b3.ra2_in1k'
    ]

    for model_name in model_names:
        # print_model_structure(model_name)  # Uncomment to print model structure
        print_model_data_config(model_name)  # Uncomment to print model data config

if __name__ == "__main__":
    main()