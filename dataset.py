import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

class OxfordPetsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.masks = []
        
        # Get paths for images and trimaps (masks)
        image_dir = os.path.join(data_dir, "images")
        mask_dir = os.path.join(data_dir, "annotations", "trimaps")
        
        # Verify directories exist
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")
        if not os.path.exists(mask_dir):
            raise ValueError(f"Trimap directory not found: {mask_dir}")
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Match each image with its trimap
        for img_name in image_files:
            base_name = os.path.splitext(img_name)[0]
            img_path = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir, f"{base_name}.png")
            
            if os.path.exists(mask_path):
                self.images.append(img_path)
                self.masks.append(mask_path)
        
        print(f"Loaded {len(self.images)} image-mask pairs")
        
        if len(self.images) == 0:
            raise ValueError("No valid image-mask pairs found in the dataset")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        
        # Apply transforms to image
        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            image = transform(image)
        
        # Process mask
        mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask)
        mask = torch.from_numpy(np.array(mask))
        
        # Convert mask to one-hot encoding (trimap has values 1,2,3)
        mask = mask - 1  # Convert to 0,1,2
        mask = torch.nn.functional.one_hot(mask.long(), num_classes=3)
        mask = mask.permute(2, 0, 1).float()  # Change to channels first format
            
        return image, mask 