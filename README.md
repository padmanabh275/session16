# Pet Segmentation with UNet

This project implements a UNet-based image segmentation model for the Oxford-IIIT Pet Dataset. It includes multiple model variants and a Gradio web interface for easy testing.

## Features

- Four UNet variants:
  1. MaxPool + Transpose Conv + BCE Loss
  2. MaxPool + Transpose Conv + Dice Loss
  3. Strided Conv + Transpose Conv + BCE Loss
  4. Strided Conv + Transpose Conv + Dice Loss
- Automatic dataset download and preparation
- PyTorch Lightning training pipeline
- Mixed precision training
- Gradio web interface
- Automatic model selection based on IoU score

## Installation

1. Clone the repository:
bash
git clone <repository-url>
cd pet-segmentation
2. Install dependencies:
bash
pip install -r requirements.txt

## Dataset

The project uses the Oxford-IIIT Pet Dataset. To download and prepare the dataset:
bash
python data.py

This will:
- Download the dataset (~800MB)
- Extract images and annotations
- Organize the files in the correct structure

## Project Structure
pet-segmentation/
├── app.py # Gradio web interface
├── config.py # Configuration settings
├── data.py # Dataset download and preparation
├── dataset.py # PyTorch dataset implementation
├── models.py # UNet model implementations
├── train.py # Training script
└── requirements.txt # Project dependencies

## Training

To train all model variants:
bash
python train.py

The training process:
- Trains four different UNet variants for 3 epochs (quick testing setup)
- Uses mixed precision training (FP16) with Tensor Core optimization
- Saves checkpoints for best models
- Tracks multiple metrics (Loss, IoU, Dice Score)
- Automatically selects the best model based on IoU score

Training configurations in `config.py`:
- Image Size: 256x256
- Batch Size: 16
- Number of Epochs: 3 (for quick testing)
- Learning Rate: 1e-4

## Model Architecture

The UNet architecture includes:
- Configurable downsampling (MaxPool or Strided Conv)
- Skip connections
- Transpose convolutions for upsampling
- Choice of loss functions (BCE or Dice)
- Batch normalization and ReLU activation

## Training Details

- Batch Size: 16
- Image Size: 256x256
- Learning Rate: 1e-4
- Optimizer: Adam
- Mixed Precision: FP16 when GPU available
- Gradient Clipping: 0.5
- Gradient Accumulation: 2 batches
- Tensor Core Optimization: High precision mode

## Hardware Optimization

- CUDA Tensor Core utilization
- Multi-worker data loading
- Persistent workers
- Prefetch factor: 2
- PIN memory for GPU transfers
- Mixed precision training (FP16)

## Requirements

- Python 3.7+
- PyTorch
- PyTorch Lightning
- torchvision
- Gradio
- PIL
- numpy
- tqdm

## Web Interface

To launch the web interface with the best performing model:
bash
python app.py


This will start a Gradio interface where you can:
- Upload pet images
- Get instant segmentation results
- View the segmentation mask

## Note on Current Setup

The current configuration uses 3 epochs for quick testing and development purposes. For production-quality results:
1. Change NUM_EPOCHS in config.py from 3 to 50
2. Expect longer training times
3. Expect better segmentation performance

#Training Results
![training_results_20250220_013531.xlsx]

## License

MIT

## Acknowledgments

- Oxford-IIIT Pet Dataset
- UNet Architecture Paper
- PyTorch Lightning
