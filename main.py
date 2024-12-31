import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

# from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from geoclip.model.GeoCLIP import GeoCLIP
from geoclip.train.train import train
from geoclip.train.eval import eval_images
from geoclip.train.dataloader import (
    GeoDataLoader,
    TaipeiDataLoader,
    img_train_transform,
    img_val_transform,
)

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


def main():
    parser = argparse.ArgumentParser(description="训练 GeoCLIP 模型（自定义数据集）")
    parser.add_argument("--dataset_csv", type=str, help="数据集的 CSV 文件路径")
    parser.add_argument("--dataset_folder", type=str, help="包含图像的数据集文件夹路径")
    parser.add_argument("--batch_size", type=int, default=4, help="训练的批次大小")
    parser.add_argument("--num_epochs", type=int, default=500, help="训练的轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--queue_size", type=int, default=4096, help="队列的大小")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="保存检查点和日志的目录"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="用于训练的设备（cuda 或 cpu）"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="数据加载的工作线程数"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="训练集所占比例（0-1之间）"
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = GeoDataLoader(
        dataset_file="./datasets/with_angle/train/taipei.csv",
        dataset_folder="./datasets/with_angle/train",
        transform=img_train_transform(),
    )
    val_ds = GeoDataLoader(
        dataset_file="./datasets/with_angle/val/taipei.csv",
        dataset_folder="./datasets/with_angle/val",
        transform=img_val_transform(),
    )

    # train_ds = TaipeiDataLoader(
    #     dataset_file="./datasets/taipei_2/train/taipei.csv",
    #     dataset_folder="./datasets/taipei_2/train",
    #     transform=img_train_transform(),
    # )
    # val_ds = TaipeiDataLoader(
    #     dataset_file="./datasets/taipei_2/val/taipei.csv",
    #     dataset_folder="./datasets/taipei_2/val",
    #     transform=img_val_transform(),
    # )

    train_dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        # drop_last=True,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=args.num_workers,
        # drop_last=True,
    )

    # rotation_criterion = TransformedGridLoss(use_cuda=True, geometric_model='affin')
    TIMEZONE_TW = timezone(timedelta(hours=8))
    dt = datetime.now(TIMEZONE_TW)
    dt = dt.strftime("%Y%m%d")

    output = Path(args.output_dir) / f"{dt}"
    print(f"checking output dir: {output}")
    if not output.exists():
        print(f"creating output dir: {output}")
        os.makedirs(output)

    writer = SummaryWriter(str(output))

    gps_gallery_path = os.path.join("./datasets/with_angle/", "gps_gallery.csv")
    if not os.path.exists(gps_gallery_path):
        all_pose = []
        bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc="Creating pose gallery",
        )
        for _, (_, coordinates, heading) in bar:
            for coord in coordinates:
                all_pose.append(coord.detach().cpu().numpy())
        df = pd.DataFrame(all_pose, columns=["LAT", "LON"])
        df.to_csv(gps_gallery_path, index=False)

        # all_pose = np.concatenate(all_pose, dim=0)
        # df = pd.DataFrame(
        #     {
        #         "LAT": all_pose[:, 0].tolist(),
        #         "LON": all_pose[:, 1].tolist(),
        #         "HEAD": all_pose[:, 2].tolist()
        #     }
        # )

    model = GeoCLIP(from_pretrained=False, queue_size=args.queue_size)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = None
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, args.num_epochs + 1):
        loss = train(
            train_dataloader,
            model,
            optimizer,
            epoch,
            args.num_epochs,
            args.batch_size,
            device,
            scaler,
            scheduler,
            criterion,
        )

        # writer.add_scalar("loss/total_loss", epoch_total_loss, epoch)
        # writer.add_scalar("loss/constrastive_loss", epoch_contrastive_loss, epoch)
        writer.add_scalar("Train/Loss", loss, epoch)

        mean_dist_error = eval_images(val_dataloader, model, device)

        writer.add_scalar("Mean Distance Errors (pixel)", mean_dist_error, epoch)

        checkpoint_path = os.path.join(output, f"geoclip_{dt}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        if scheduler is not None:
            scheduler.step()
    writer.close()


if __name__ == "__main__":
    main()
