import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py

from geoclip.model.GeoCLIP import GeoCLIP
from geoclip.model.HomoCLIP import HomoCLIP
from geoclip.train.train import train
from geoclip.train.eval import eval_images
from geoclip.train.dataloader import (
    GeoDataLoader,
    TaipeiDataLoader,
    ImgGalleryDataloader,
    img_train_transform,
    img_val_transform,
)

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


def main():
    parser = argparse.ArgumentParser(description="训练 GeoCLIP 模型（自定义数据集）")
    parser.add_argument("--batch_size", type=int, default=16, help="训练的批次大小")
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
    parser.add_argument(
        "--backbone", type=str, default='./output/20241227/geoclip_20241227.pth', help="backbone"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="resume model"
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = TaipeiDataLoader(
        dataset_file="./datasets/taipei_gallery/train/taipei_30000.csv",
        dataset_folder="./datasets/taipei_gallery/train",
        transform=img_train_transform(),
    )
    val_ds = TaipeiDataLoader(
        dataset_file="./datasets/taipei_gallery/val/taipei_12000.csv",
        dataset_folder="./datasets/taipei_gallery/val",
        transform=img_val_transform(),
    )

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
        batch_size=args.batch_size,
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

    gps_gallery_path = os.path.join("./datasets/taipei_gallery/", "gps_gallery.csv")
    if not os.path.exists(gps_gallery_path):
        all_pose = []
        bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc="Creating pose gallery",
        )
        for idx, (ref_imgs, rot_imgs, coords, headings) in bar:
            pose = np.concatenate((coords, headings), axis=1)
            pose = pose.tolist()
            # all_pose.append(pose.reshape(-1, pose.shape[-1]))
            all_pose.append(pose)

        all_pose = np.vstack(all_pose)
        data = np.array(all_pose).tolist()
        df = pd.DataFrame(data, columns=["LAT", "LON", "HEAD_SIN", "HEAD_COS"])
        df.to_csv(gps_gallery_path, index=False)
        print(f"gps gallery csv is created successfully")
        del all_pose

    geoclip = GeoCLIP(from_pretrained=True, queue_size=args.queue_size)
    if args.backbone:
            geoclip.load_state_dict(torch.load(args.backbone, weights_only=True))

    geoclip.to(device)
    # hdf5_file = os.path.join("./datasets/taipei_gallery/", "img_gallery.h5")
    # if not os.path.exists(hdf5_file):
    #     bar = tqdm(
    #         enumerate(train_dataloader),
    #         total=len(train_dataloader),
    #         desc="Creating image gallery (hdf5 database)",
    #     )
    #     sample_ref_image, sample_rot_image, coord, heading = train_ds[0]
    #     channels, img_height, img_width = sample_ref_image.shape
    #
    #     with h5py.File(hdf5_file, 'w') as hdf:
    #         image_dataset = hdf.create_dataset(
    #             'images',
    #             shape=(len(train_ds), channels, img_height, img_width),
    #             dtype=np.float32,
    #             compression='gzip',
    #             chunks=(100, channels, img_height, img_width)
    #         )
    #
    #         label_dataset = hdf.create_dataset(
    #             'labels',
    #             shape=(len(train_ds), 2),
    #             dtype=np.float32
    #         )
    #
    #         index = 0
    #         for _, (ref_imgs, rot_imgs, coordinates, headings) in bar:
    #             image_dataset[index:index+args.batch_size] = ref_imgs
    #             label_dataset[index:index+args.batch_size] = headings
    #             index += args.batch_size
    #     print(f"HDF5 file '{hdf5_file}' created successfully with {len(train_ds)} images.")

    # img_gallery_ds = ImgGalleryDataloader(hdf5_file)
    model = HomoCLIP()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = None
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, args.num_epochs + 1):
        train_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        model.train()
        geoclip.eval()
        total_loss = 0.0

        for i ,(ref_imgs, rot_imgs, coordinates, gt_headings) in train_bar:
            ref_imgs = ref_imgs.to(device)
            rot_imgs = rot_imgs.to(device)
            coordinates = coordinates.to(device)

            with torch.no_grad():
                rot_img_embeddings = geoclip.image_encoder.preprocess_image(rot_imgs)
                ref_imgs_embeddings = geoclip.image_encoder.preprocess_image(ref_imgs)

            # Forward pass
            with torch.autocast(device_type="cuda"): 
                logits_img_gps = model(rot_img_embeddings, ref_imgs_embeddings)
                exit()
                # Compute the loss
                loss = criterion(logits_img_gps, targets_img_gps)

            # Backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            items += 1

            bar.set_description(f"Epoch {epoch}/{total_epochs}")

        if scheduler is not None:
            scheduler.step()

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
