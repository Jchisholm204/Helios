import random
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Union, Any
import matplotlib.pyplot as plt


class ExperimentLogger:
    """
    A class for managing, saving, loading, and visualizing training and
    validation metrics for a deep learning experiment.
    """

    def __init__(self, experiment_name: str, log_root_dir: str = 'logs'):
        """
        Initializes the logger, sets up the experiment directory, and initializes 
        the internal log dictionary.

        Args:
            experiment_name (str): A descriptive name for the experiment.
            log_root_dir (str): The base directory where all logs will be stored.
        """
        self.experiment_name = experiment_name
        self.log_root_dir = log_root_dir
        self.log_dir = self._setup_log_directory()
        self.log_file_path = os.path.join(self.log_dir, 'metrics.json')
        self.plot_file_path = os.path.join(self.log_dir, 'loss_curves.png')
        self.hparams_file_path = os.path.join(self.log_dir, 'hparams.json')

        # Internal dictionary to hold all metrics
        self.log_data: Dict[str, List[Union[float, int]]] = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        print(f"Logger initialized. Log directory: {self.log_dir}")

    def _setup_log_directory(self) -> str:
        """
        Creates a unique log directory using the experiment name and a timestamp.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{self.experiment_name}_{timestamp}"
        log_dir = os.path.join(self.log_root_dir, dir_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def get_log_path(self) -> str:
        """
        Returns the path to the current experiment's logging directory.
        Useful for saving models, checkpoints, etc.
        """
        return self.log_dir

    def save_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Saves the current epoch's metrics to the internal log dictionary.

        Args:
            epoch (int): The current epoch number.
            metrics (Dict[str, float]): A dictionary containing metric names 
                                         (e.g., 'train_loss', 'val_loss') and values.
        """
        self.log_data['epoch'].append(epoch)
        for key, value in metrics.items():
            # Automatically create metric list if it doesn't exist
            if key not in self.log_data:
                self.log_data[key] = []

            # Ensure all previous entries have a placeholder if this metric is new
            # This is robust logging, though standard practice is to log all metrics together.
            if len(self.log_data[key]) < len(self.log_data['epoch']) - 1:
                # Fill missing previous epochs with None
                self.log_data[key].extend(
                    [None] * (len(self.log_data['epoch']) - 1 - len(self.log_data[key])))

            self.log_data[key].append(value)

        # Save to JSON after updating metrics
        self.save_log_to_json()

    def save_log_to_json(self):
        """
        Writes the internal log_data dictionary to a JSON file.
        """
        try:
            with open(self.log_file_path, 'w') as f:
                json.dump(self.log_data, f, indent=4)
        except Exception as e:
            print(f"ERROR: Could not save log data to JSON: {e}")

    def load_log_from_json(self, filepath: str):
        """
        Loads the metrics from a JSON file into the internal log_data.
        """
        try:
            with open(filepath, 'r') as f:
                self.log_data = json.load(f)
                print(f"Log data successfully loaded from: {filepath}")
        except FileNotFoundError:
            print(f"ERROR: Log file not found at {filepath}")
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to decode JSON log file: {e}")

    def save_hyperparameters(self, hparams: Dict[str, Any]):
        """
        Saves the experiment's hyperparameters to a separate JSON file.
        """
        try:
            with open(self.hparams_file_path, 'w') as f:
                json.dump(hparams, f, indent=4)
            print(f"Hyperparameters saved to: {self.hparams_file_path}")
        except Exception as e:
            print(f"ERROR: Could not save hyperparameters: {e}")

    def plot_loss_curves(self, show_plot: bool = True, save_plot: bool = True):
        """
        Plots the training and validation loss curves.

        Args:
            show_plot (bool): If True, displays the plot in a new window.
            save_plot (bool): If True, saves the plot to the log directory.
        """
        epochs = self.log_data.get('epoch', [])
        train_loss = self.log_data.get('train_loss', [])
        val_loss = self.log_data.get('val_loss', [])

        if not epochs:
            print("No epochs logged. Cannot plot curves.")
            return

        plt.figure(figsize=(10, 6))

        if train_loss:
            plt.plot(epochs, train_loss, label='Training Loss',
                     marker='o', linestyle='-', color='blue')
        if val_loss:
            plt.plot(epochs, val_loss, label='Validation Loss',
                     marker='x', linestyle='--', color='red')

        plt.title(f'Loss Curves for Experiment: {self.experiment_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_plot:
            try:
                plt.savefig(self.plot_file_path)
                print(f"Loss curve plot saved to: {self.plot_file_path}")
            except Exception as e:
                print(f"ERROR: Could not save plot file: {e}")

        if show_plot:
            plt.show()


def simulate_training(num_epochs: int):
    """
    Simulates a training loop using the ExperimentLogger.
    """

    # 1. Configuration and Initialization
    EXP_NAME = "CLIP_COCO_FineTune"
    hparams = {
        "model": "clip-vit-base-patch32",
        "learning_rate": 1e-5,
        "batch_size": 32,
        "optimizer": "AdamW"
    }

    logger = ExperimentLogger(experiment_name=EXP_NAME)

    # Save Hyperparameters immediately
    logger.save_hyperparameters(hparams)

    # Check the model save path
    model_save_path = os.path.join(logger.get_log_path(), "best_model.pth")
    print(f"Model checkpoint will be saved to: {model_save_path}")

    # 2. Simulated Training Loop
    print("\n--- Starting Simulated Training ---")

    current_train_loss = 2.5
    current_val_loss = 2.8

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}...")

        # Simulate loss reduction and noise
        current_train_loss = max(
            0.1, current_train_loss * 0.95 + random.uniform(-0.05, 0.05))
        current_val_loss = max(0.1, current_val_loss *
                               0.96 + random.uniform(-0.03, 0.03))

        # Simulate accuracy calculation
        train_acc = 100 - (current_train_loss * 10)
        val_acc = 100 - (current_val_loss * 12)

        # 3. Log Metrics
        metrics = {
            'train_loss': current_train_loss,
            'val_loss': current_val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
        logger.save_metrics(epoch, metrics)

        # In a real scenario, you would save your model checkpoint here
        # torch.save(model.state_dict(), model_save_path)

        time.sleep(0.1)  # Simulate training time

    print("--- Training Complete ---")

    # 4. Display and Save Plot
    logger.plot_loss_curves(show_plot=False, save_plot=True)  # Save only

    # 5. Demonstration of Loading
    print("\n--- Demonstrating Log Loading ---")
    new_logger = ExperimentLogger("Load_Test")
    new_logger.load_log_from_json(logger.log_file_path)
    print(f"Total epochs loaded: {len(new_logger.log_data.get('epoch', []))}")

    # Display the loaded plot (only show, don't save again)
    new_logger.plot_loss_curves(show_plot=True, save_plot=False)


if __name__ == '__main__':
    simulate_training(num_epochs=15)
