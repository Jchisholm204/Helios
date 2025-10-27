# SnoutNet Testing and Visualization Guide

This guide covers the testing and visualization scripts for the SnoutNet model.

## Files Created

### 1. `test.py` - Model Evaluation Script
Calculates comprehensive localization accuracy statistics on the test dataset.

### 2. `show.py` - Prediction Visualization Script  
Visualizes model predictions on test images with ground truth comparisons.

## Usage

### Testing Model Performance (`test.py`)

#### Basic Usage
```bash
cd /home/jacob/Documents/Helios/ELEC475/Lab2
source venv/bin/activate
python src/test.py <model_path>
```

#### Example
```bash
python src/test.py checkpoints/snoutnet_20251026_184103/best_model.pth
```

#### Output
- **Console**: Detailed accuracy statistics
- **Plot**: Distance distribution histogram (`test_results_*.png`)
- **File**: Detailed results (`test_results_*.txt`)

#### Statistics Calculated
- **Euclidean Distance**: Min, Max, Mean, Median, Std Dev
- **Percentiles**: 25th, 75th, 90th, 95th
- **Performance Metrics**: Inference time, throughput
- **Performance Interpretation**: Excellent/Good/Fair/Poor classification

### Visualizing Predictions (`show.py`)

#### Basic Usage
```bash
python src/show.py <model_path> [options]
```

#### Options
- `--num_samples N`: Number of samples to visualize (default: 12)
- `--single INDEX`: Visualize single sample at given index
- `--save PATH`: Save visualization to file

#### Examples

**Grid visualization (multiple samples):**
```bash
python src/show.py checkpoints/snoutnet_20251026_184103/best_model.pth --num_samples 8
```

**Single sample visualization:**
```bash
python src/show.py checkpoints/snoutnet_20251026_184103/best_model.pth --single 42
```

**Save visualization:**
```bash
python src/show.py checkpoints/snoutnet_20251026_184103/best_model.pth --save my_predictions.png
```

#### Visualization Features
- **Grid Layout**: Multiple samples in organized grid
- **Ground Truth**: Green circles marking actual nose locations
- **Predictions**: Red crosses marking predicted locations
- **Error Vectors**: Blue dashed lines showing prediction errors
- **Zoomed Views**: Detailed view of nose area for single samples
- **Error Statistics**: Per-sample and summary error metrics

## Sample Results

### Test Results Example
```
SNOUTNET LOCALIZATION ACCURACY STATISTICS
============================================================
Total test samples: 698
Inference time: 0.08 seconds
Average inference time: 0.12 ms per sample
Throughput: 8305.2 samples/second

Euclidean Distance Statistics (pixels):
  Minimum:     25.15
  Maximum:     2359.55
  Mean:        225.63
  Median:      220.41
  Std Dev:     121.00

Percentiles:
  25th:        160.65
  75th:        274.56
  90th:        335.74
  95th:        381.32

Performance Interpretation:
  ❌ POOR: Mean error > 200 pixels
```

### Visualization Summary Example
```
Visualization Summary:
Number of samples: 4
Mean error: 223.08 pixels
Min error: 160.72 pixels
Max error: 321.73 pixels
Std error: 62.57 pixels
```

## Performance Interpretation

### Distance Error Categories
- **🎯 EXCELLENT**: Mean error < 50 pixels
- **✅ GOOD**: Mean error < 100 pixels  
- **⚠️ FAIR**: Mean error < 200 pixels
- **❌ POOR**: Mean error > 200 pixels

### Current Model Performance
Based on test results, the current model shows:
- **Mean Error**: ~226 pixels (POOR performance)
- **Range**: 25-2359 pixels (high variance)
- **Throughput**: ~8300 samples/second (very fast inference)

## File Outputs

### Generated Files
1. **`test_results_*.png`**: Distance distribution plots
2. **`test_results_*.txt`**: Detailed numerical results
3. **`predictions_grid_*.png`**: Grid visualization of predictions
4. **`single_prediction_*.png`**: Single sample detailed view

## Technical Details

### Model Loading
- Handles dynamic FC layer initialization
- Proper device placement (CUDA/CPU)
- Checkpoint validation and error handling

### Distance Calculation
- Euclidean distance: `√((x₁-x₂)² + (y₁-y₂)²)`
- Pixel-level accuracy measurement
- Statistical analysis with percentiles

### Visualization
- Image denormalization for proper display
- Coordinate system handling (image vs matplotlib)
- Error vector visualization
- Zoomed detail views

## Troubleshooting

### Common Issues
1. **Model not found**: Check path to checkpoint file
2. **Device mismatch**: Ensure CUDA is available if using GPU
3. **Index out of range**: Check sample indices are within dataset bounds

### Requirements
- PyTorch
- matplotlib
- PIL (Pillow)
- numpy
- pandas

All dependencies are included in the virtual environment.
