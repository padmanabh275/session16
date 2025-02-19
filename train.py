import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import OxfordPetsDataset
from models import UNet
from config import ModelConfig
import os
import json
import pandas as pd
from datetime import datetime

# Set float32 matmul precision for better performance on Tensor Cores
torch.set_float32_matmul_precision('high')

def train_model(model_config, use_strided_conv=False, loss_fn="bce"):
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((model_config.IMAGE_HEIGHT, model_config.IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    dataset = OxfordPetsDataset(model_config.DATA_PATH, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Calculate optimal number of workers and check platform
    if os.name == 'nt':  # Windows
        num_workers = 0  # Use single process on Windows to avoid shared memory issues
    else:  # Linux/Mac
        # Calculate optimal number of workers
        total_cpu_count = os.cpu_count() or 1
        recommended_workers = 19
        # Use the minimum of: recommended workers, CPU count * 4, or 32
        num_workers = min(recommended_workers, total_cpu_count * 4, 32)
        print(f"Using {num_workers} workers for data loading")
    
    # Create dataloaders with platform-specific settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=model_config.BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=model_config.BATCH_SIZE,
        num_workers=num_workers,  # Same number of workers for validation
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Create model
    model = UNet(
        n_channels=model_config.NUM_CHANNELS,
        n_classes=model_config.NUM_CLASSES,
        use_strided_conv=use_strided_conv,
        loss_fn=loss_fn
    )
    
    # Create checkpoint callbacks for different metrics
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename=f'unet-{use_strided_conv}-{loss_fn}-loss',
            save_top_k=1,
            mode='min'
        ),
        ModelCheckpoint(
            monitor='val_iou',
            dirpath='checkpoints/',
            filename=f'unet-{use_strided_conv}-{loss_fn}-iou',
            save_top_k=1,
            mode='max'
        ),
        ModelCheckpoint(
            monitor='val_dice',
            dirpath='checkpoints/',
            filename=f'unet-{use_strided_conv}-{loss_fn}-dice',
            save_top_k=1,
            mode='max'
        )
    ]
    
    # Create trainer with validation and optimization settings
    trainer = pl.Trainer(
        max_epochs=model_config.NUM_EPOCHS,
        callbacks=callbacks,  # Updated to use multiple callbacks
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16 if torch.cuda.is_available() else 32,
        val_check_interval=0.25,
        gradient_clip_val=0.5,
        accumulate_grad_batches=2,
    )
    
    # Train and validate model
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    test_results = trainer.test(model, val_loader)[0]
    
    # Combine metrics from callbacks and test results
    best_metrics = {
        'loss': callbacks[0].best_model_score.item(),
        'iou': callbacks[1].best_model_score.item(),
        'dice': callbacks[2].best_model_score.item(),
        'checkpoint_path': callbacks[1].best_model_path,  # Save path of best IoU model
        'test_iou': test_results['test_iou'],
        'test_dice': test_results['test_dice']
    }
    
    print(f"\nFinal Results for {use_strided_conv}-{loss_fn}:")
    print(f"Best Val IoU: {best_metrics['iou']:.4f}")
    print(f"Test IoU: {best_metrics['test_iou']:.4f}")
    print(f"Best Val Dice: {best_metrics['dice']:.4f}")
    print(f"Test Dice: {best_metrics['test_dice']:.4f}")
    
    return model, best_metrics

if __name__ == "__main__":
    config = ModelConfig()
    
    # Enable deterministic training for reproducibility
    pl.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create results dictionary and lists to store all metrics
    results = {}
    detailed_results = []
    best_overall_model = None
    best_overall_iou = 0
    
    for name, params in {
        "MP_TR_BCE": (False, "bce"),
        "MP_TR_DICE": (False, "dice"),
        "STRCONV_TR_BCE": (True, "bce"),
        "STRCONV_UPS_DICE": (True, "dice")
    }.items():
        print(f"\nTraining {name}...")
        model, metrics = train_model(config, use_strided_conv=params[0], loss_fn=params[1])
        results[name] = metrics
        
        # Store detailed results
        detailed_results.append({
            'Model': name,
            'Architecture': 'Strided Conv' if params[0] else 'MaxPool',
            'Loss Function': params[1].upper(),
            'Best Val Loss': metrics['loss'],
            'Best Val IoU': metrics['iou'],
            'Best Val Dice': metrics['dice'],
            'Test IoU': metrics['test_iou'],
            'Test Dice': metrics['test_dice'],
            'Checkpoint Path': metrics['checkpoint_path']
        })
        
        # Track the best model
        if metrics['iou'] > best_overall_iou:
            best_overall_iou = metrics['iou']
            best_overall_model = {
                'name': name,
                'checkpoint_path': metrics['checkpoint_path']
            }
    
    # Save best model info for app.py
    with open('best_model.json', 'w') as f:
        json.dump(best_overall_model, f)
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(detailed_results)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f'training_results_{timestamp}.xlsx'
    
    # Create Excel writer with multiple sheets
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Results table
        df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        # Configuration sheet
        config_dict = {
            'Parameter': ['Image Height', 'Image Width', 'Num Channels', 'Num Classes', 
                         'Batch Size', 'Num Epochs', 'Learning Rate'],
            'Value': [config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.NUM_CHANNELS, 
                     config.NUM_CLASSES, config.BATCH_SIZE, config.NUM_EPOCHS, 
                     config.LEARNING_RATE]
        }
        pd.DataFrame(config_dict).to_excel(writer, sheet_name='Configuration', index=False)
        
        # Best Model Info
        best_model_df = pd.DataFrame([{
            'Best Model': best_overall_model['name'],
            'Best IoU Score': best_overall_iou,
            'Checkpoint Path': best_overall_model['checkpoint_path']
        }])
        best_model_df.to_excel(writer, sheet_name='Best Model', index=False)
    
    print(f"\nResults saved to {excel_filename}")
    
    # Print comparison of all models
    print("\nFinal Comparison of All Models:")
    print("=" * 50)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"Best Val Loss: {metrics['loss']:.4f}")
        print(f"Best Val IoU: {metrics['iou']:.4f}")
        print(f"Test IoU: {metrics['test_iou']:.4f}")
        print(f"Best Val Dice: {metrics['dice']:.4f}")
        print(f"Test Dice: {metrics['test_dice']:.4f}")
    
    print(f"\nBest Overall Model: {best_overall_model['name']}")
    print(f"Best Overall IoU: {best_overall_iou:.4f}") 