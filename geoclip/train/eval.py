from os import walk
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm


def distance_accuracy(targets, preds, dis=2500, gps_gallery=None):
    total = len(targets)
    correct = 0
    avg_distance_error = 0
    accuracy = 0

    for i in range(total):
        pred_coord = gps_gallery[preds[i]].detach().cpu().numpy()
        true_coord = targets[i]
        distance = np.linalg.norm(pred_coord - true_coord)
        avg_distance_error += distance
        # if distance <= dis:
        #     correct += 1

    avg_distance_error /= total
    # accuracy = correct / total
    return accuracy, avg_distance_error

# def mae_loss(preds, targets, gps_gallery):
#     assert len(preds) == len(targets)
#     return np.sum(np.abs(gps_gallery[preds.detach().cpu().numpy()] - targets))

def eval_images(val_dataloader, model, device="cpu"):
    model.eval()

    all_gps = []
    for _, gps, _ in tqdm(val_dataloader, desc="Evaluation"):
        all_gps.append(gps)
    all_gps = torch.cat(all_gps, dim=0).to(device)
    model.set_gps_gallery(all_gps)

    preds = []
    targets = []

    gps_gallery = model.gps_gallery

    with torch.no_grad():
        for ref_imgs, labels, _ in tqdm(val_dataloader, desc="Evaluation"):
            labels = labels.cpu().numpy()
            imgs = ref_imgs.to(device)

            # Get predictions (probabilities for each location based on similarity)
            with torch.autocast(device_type="cuda"): 
                logits_per_image = model(imgs, gps_gallery)
            probs = logits_per_image.softmax(dim=-1)
            
            # Predict gps location with the highest probability (index)
            outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
            
            preds.append(outs)
            targets.append(labels)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    model.train()

    # distance_thresholds = [2500, 750, 200, 25, 1] # km
    # accuracy_results = {}
    # for dis in distance_thresholds:
    #     acc, avg_distance_error = distance_accuracy(targets, preds, dis, gps_gallery)
    #     print(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")
    #     accuracy_results[f'acc_{dis}_km'] = acc
    acc, avg_dist_error = distance_accuracy(targets, preds, 200, gps_gallery)
    # avg_dist_error = F.l1_loss(preds, targets)

    return avg_dist_error

