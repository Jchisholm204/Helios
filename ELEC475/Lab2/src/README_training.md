# SnoutNet Training Guide

This directory contains the training script for the SnoutNet model, which performs pet nose centerpoint detection using regression.

## Files

- `train.py` - Main training script
- `model.py` - SnoutNet model definition
- `dataloader.py` - Dataset and data loading utilities
- `test_model.py` - Model testing script

## Quick Start

### 1. Activate Virtual Environment
```bash
cd /home/jacob/Documents/Helios/ELEC475/Lab2
source venv/bin/activate
```

### 2. Run Training
```bash
cd src
python train.py
```

## Training Configuration

The training script uses the following default configuration:

- **Model**: SnoutNet (3 conv layers + 2 FC layers)
- **Input Size**: 227x227x3
- **Output**: 2D coordinates (x, y)
- **Loss Function**: MSE (Mean Squared Error) for regression
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 50
- **Data Augmentation**: None (as specified)

## Dataset

- **Training**: `oxford-iiit-pet-noses/train_noses.txt` (5,494 samples)
- **Validation**: `oxford-iiit-pet-noses/test_noses.txt` (698 samples)
- **Images**: `oxford-iiit-pet-noses/images-original/images/`

## Output

The training script will create:

1. **Model Checkpoints**: Saved in `./checkpoints/snoutnet_YYYYMMDD_HHMMSS/`
   - `best_model.pth` - Best model based on validation loss
   - `final_model.pth` - Final model after all epochs

2. **Loss Plot**: `training_losses.png` - Visualization of training progress

3. **Console Output**: Real-time training progress and metrics

## Customization

You can modify the training parameters by editing the `config` dictionary in the `main()` function:

```python
config = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'target_size': 227,
    'num_workers': 4
}
```

## Model Architecture

SnoutNet consists of:
- 3 Convolutional layers with BatchNorm and ReLU
- 2 Fully Connected layers
- Dropout for regularization
- Output: 2D coordinates (x, y) for nose centerpoint

## Training Features

- **Regression Loss**: MSE loss for coordinate prediction
- **Validation**: Uses test partition for validation (not for optimization)
- **Learning Rate Scheduling**: Reduces LR on plateau
- **Model Saving**: Saves best and final models
- **Progress Tracking**: Real-time loss monitoring
- **Loss Visualization**: Automatic plotting of training curves

## Requirements

- PyTorch
- torchvision
- matplotlib
- pandas
- PIL (Pillow)
- numpy

All dependencies are included in the virtual environment.
