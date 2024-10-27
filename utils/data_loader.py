#Completed? ye
import os
import numpy as np
from src.features_extraction import extract_mfcc

def load_data(data_dir, labels_file):
    # Load audio file paths and corresponding labels
    audio_files = []
    labels = []
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip commented lines
            if line.startswith('#') or not line.strip():
                continue
            file_name, label = line.strip().split(',')
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                audio_files.append(file_path)
                labels.append(label)
            else:
                print(f"Warning: File {file_name} does not exist, skipping this entry")

    # Extract MFCC features from each audio file
    X = np.array([extract_mfcc(file) for file in audio_files])
    
    y = np.array(labels)
    return X, y
