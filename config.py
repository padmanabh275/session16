from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    # Model configurations
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    NUM_CHANNELS = 3
    NUM_CLASSES = 3  # Changed to 3 since Oxford Pet trimap has 3 classes
    
    # Training configurations
    BATCH_SIZE = 16
    NUM_EPOCHS = 3    # Changed from 50 to 3 for quick testing
    LEARNING_RATE = 1e-4
    
    # Dataset path - matching data.py
    DATA_PATH = "Oxford-IIIT Pet Dataset"
    
    # Dataset path - assuming the Oxford-IIIT Pet Dataset structure
    DATA_PATH_OLD = "Oxford-IIIT Pet Dataset"  # or provide the full path where you downloaded the dataset 