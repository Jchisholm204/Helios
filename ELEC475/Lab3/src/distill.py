# import torch
import torch
# Dataloader import
from data import PascalVOCSegmentation
# Models
from segmodel import MobileNetV3_SegNet
from fcnresnet import FCNResNetWrapper
# Response based learning
import distill_response_loss as drl
import distill_features as df
from plotter import plot_history
from datetime import datetime
import os

EPOCHS = 200
LEARNING_RATE = 0.003
LEARNING_DECAY = 0.0003
BATCH_SIZE = 2
WORKERS = 4
# Feature Based
TEMP = 1
ALPHA = 1
BETA = 0
# Response Based
# TEMP = 0.9
# ALPHA = 0.9
# BETA = 0.2


def load_model(checkpoint_path: str = None, device='cuda') -> torch.nn.Module:
    print("Loading Model...")
    segnet = MobileNetV3_SegNet()
    if checkpoint_path:
        print(f"Loading Checkpoint {checkpoint_path}")
        segnet.load_state_dict(torch.load(checkpoint_path))
    segnet.to(device)
    n_train_params = sum(p.numel()
                         for p in segnet.parameters() if p.requires_grad)
    print(f"Loaded Model on {device} with {n_train_params} parameters")
    return segnet


if __name__ == "__main__":
    data = PascalVOCSegmentation(batch_size=BATCH_SIZE, n_workers=WORKERS)
    train_loader = data.get_train_loader()
    val_loader = data.get_train_loader()
    save_path = f"distill_{datetime.now().strftime("%Y%M%d_%H-%M-%S")}"
    os.mkdir(save_path)
    print(f"Created {save_path}")
    # model = load_model('./distill_20252808_18-28-31/final.pth')
    model = load_model()
    # model = load_model('./distill_20253708_20-37-14/final.pth')
    teacher = FCNResNetWrapper().to('cuda')
    history = df.distill_model(model, teacher, train_loader, val_loader,
                               ALPHA, BETA, TEMP,
                               int(EPOCHS), LEARNING_RATE, LEARNING_DECAY,
                               save_path=save_path)

    plot_history(history, save_path=save_path)
