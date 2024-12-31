import torch
from torch import nn
import torch.nn.functional as F
# from torch.cuda.amp import autocast as autocast, GradScaler
import numpy as np
from tqdm import tqdm
import cv2


def train(train_dataloader, model, optimizer, epoch, total_epochs, batch_size, device, scaler, scheduler=None, criterion=nn.CrossEntropyLoss(), rotation_criterion=None):
    # print("Starting Epoch", epoch)

    model.train()
    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(device)

    epoch_total_loss = 0.0
    items = 0

    for i ,(ref_imgs , coordinates, gt_heading) in bar:
        # imgs = imgs.to(device)
        ref_imgs = ref_imgs.to(device)
        # rot_imgs = rot_imgs.to(device)
        coordinates = coordinates.to(device)
        # gt_orientations = gt_orientations.to(device)

        gps_queue = model.get_gps_queue()
        optimizer.zero_grad()

        # Append GPS Queue & Queue Update
        pose_all = torch.cat([coordinates, gps_queue], dim=0).to(device)
        model.dequeue_and_enqueue(coordinates)

        # Forward pass
        with torch.autocast(device_type="cuda"): 
            logits_img_gps = model(ref_imgs, pose_all)

            # Compute the loss
            loss = criterion(logits_img_gps, targets_img_gps)

        # Backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        epoch_total_loss += loss.item()
        items += 1

        bar.set_description(f"Epoch {epoch}/{total_epochs}")

    if scheduler is not None:
        scheduler.step()

    return np.mean(epoch_total_loss)
