import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import JaccardIndex, Dice

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, use_strided_conv=False, loss_fn="bce"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.loss_fn = loss_fn

        self.inc = DoubleConv(n_channels, 64)
        
        # Encoder
        if use_strided_conv:
            self.down1 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                DoubleConv(128, 128)
            )
            self.down2 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                DoubleConv(256, 256)
            )
            self.down3 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                DoubleConv(512, 512)
            )
            self.down4 = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
                DoubleConv(1024, 1024)
            )
        else:
            self.down1 = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(64, 128)
            )
            self.down2 = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(128, 256)
            )
            self.down3 = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(256, 512)
            )
            self.down4 = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(512, 1024)
            )

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.up_conv1 = DoubleConv(1024, 512)
        self.up_conv2 = DoubleConv(512, 256)
        self.up_conv3 = DoubleConv(256, 128)
        self.up_conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        # Add metrics
        self.iou = JaccardIndex(task="multiclass", num_classes=n_classes)
        self.dice = Dice(num_classes=n_classes, average='macro')
        
        # Add loss functions
        if loss_fn == "bce":
            self.criterion = nn.BCEWithLogitsLoss()  # Combined sigmoid + BCE
        
        self.save_hyperparameters()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = self.up_conv1(torch.cat([x4, x], dim=1))
        x = self.up2(x)
        x = self.up_conv2(torch.cat([x3, x], dim=1))
        x = self.up3(x)
        x = self.up_conv3(torch.cat([x2, x], dim=1))
        x = self.up4(x)
        x = self.up_conv4(torch.cat([x1, x], dim=1))
        x = self.outc(x)
        return x  # Remove sigmoid here, it's included in BCEWithLogitsLoss

    def dice_loss(self, pred, target):
        smooth = 1e-5
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / 
                   (pred.sum() + target.sum() + smooth))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if self.loss_fn == "bce":
            loss = self.criterion(y_hat, y)  # Use criterion instead of F.binary_cross_entropy
        else:  # dice loss
            loss = self.dice_loss(y_hat, y)
        
        # For metrics, we need probabilities, so apply sigmoid here
        y_hat_probs = torch.sigmoid(y_hat)
        
        # Calculate metrics
        pred_masks = torch.argmax(y_hat_probs, dim=1)
        true_masks = torch.argmax(y, dim=1)
        
        iou_score = self.iou(pred_masks, true_masks)
        dice_score = self.dice(pred_masks, true_masks)
        
        # Log metrics
        self.log('train_loss', loss)
        self.log('train_iou', iou_score)
        self.log('train_dice', dice_score)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if self.loss_fn == "bce":
            loss = self.criterion(y_hat, y)
        else:
            loss = self.dice_loss(y_hat, y)
        
        # Apply sigmoid for metrics
        y_hat_probs = torch.sigmoid(y_hat)
        
        # Calculate metrics
        pred_masks = torch.argmax(y_hat_probs, dim=1)
        true_masks = torch.argmax(y, dim=1)
        
        iou_score = self.iou(pred_masks, true_masks)
        dice_score = self.dice(pred_masks, true_masks)
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_iou', iou_score)
        self.log('val_dice', dice_score)
        
        return {'val_loss': loss, 'val_iou': iou_score, 'val_dice': dice_score}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if self.loss_fn == "bce":
            loss = self.criterion(y_hat, y)
        else:
            loss = self.dice_loss(y_hat, y)
        
        # Apply sigmoid for metrics
        y_hat_probs = torch.sigmoid(y_hat)
        
        # Calculate metrics
        pred_masks = torch.argmax(y_hat_probs, dim=1)
        true_masks = torch.argmax(y, dim=1)
        
        iou_score = self.iou(pred_masks, true_masks)
        dice_score = self.dice(pred_masks, true_masks)
        
        # Log metrics with 'test_' prefix
        self.log('test_loss', loss)
        self.log('test_iou', iou_score)
        self.log('test_dice', dice_score)
        
        return {
            'test_loss': loss,
            'test_iou': iou_score,
            'test_dice': dice_score
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4) 