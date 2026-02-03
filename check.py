import pickle 
import pandas as pd
#c = pickle.load("trained_model/Sorzoi/model_best.pth")
#print(c.columns)
#print(c)
import torch

file_path = "trained_model/Sorzoi/model_best.pth" 
try:
    # Load the state dictionary
    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))

    print(f"Contents of '{file_path}':")

    # Check if it's a state_dict and print keys and shapes
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_state_dict = checkpoint['state_dict']
        print("\nKeys in state_dict (checkpoint format):")
        for key, value in model_state_dict.items():
            print(f"* {key}: {value.shape}")
    elif isinstance(checkpoint, dict):
        print("\nKeys in state_dict (weights only format):")
        for key, value in checkpoint.items():
            print(f"* {key}: {value.shape}")
    else:
        print(f"\nThe loaded object is not a dictionary. Type: {type(checkpoint)}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Ensure the file is a valid PyTorch file and the path is correct.")
