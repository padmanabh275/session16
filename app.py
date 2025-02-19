import gradio as gr
import torch
import json
from PIL import Image
import torchvision.transforms as transforms
from models import UNet
from config import ModelConfig

config = ModelConfig()

# Load information about the best model
with open('best_model.json', 'r') as f:
    best_model_info = json.load(f)

# Create and load the best model
model_name = best_model_info['name']
use_strided_conv = 'STRCONV' in model_name
loss_fn = 'dice' if 'DICE' in model_name else 'bce'

model = UNet(config.NUM_CHANNELS, config.NUM_CLASSES, use_strided_conv, loss_fn)
model.load_state_dict(torch.load(best_model_info['checkpoint_path'])['state_dict'])
model.eval()

def predict(image):
    """Make prediction using the best model"""
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])
    
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        pred = model(image)
        pred = torch.sigmoid(pred)
    
    # Convert prediction to segmentation mask
    pred = torch.argmax(pred, dim=1)
    pred = pred.squeeze().numpy()
    
    return pred

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="numpy"),
    title="Pet Segmentation with Best UNet Model",
    description=f"Using {model_name} model for segmentation"
)

if __name__ == "__main__":
    iface.launch() 