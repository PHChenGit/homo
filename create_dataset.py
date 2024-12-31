import argparse
import glob
import math
import os
from pathlib import Path
import time
import numpy as np
import random
from PIL import Image
import pandas as pd
from tqdm import tqdm
import cv2
import json

from utils import OutBoundaryException


def rotate_and_crop(image, center, crop_size, angle):
    width, height = image.size
    center_x, center_y = center

    def rotate_point(x, y, cx, cy, angle):
        rad = math.radians(angle)
        x_new = math.cos(rad) * (x - cx) - math.sin(rad) * (y - cy) + cx
        y_new = math.sin(rad) * (x - cx) + math.cos(rad) * (y - cy) + cy
        return x_new, y_new

    rotated_img = image.rotate(angle, expand=True)


def crop(image, center, crop_size):
    center_x, center_y = center
    half_crop = crop_size // 2

    corners = [
        (center_x - half_crop, center_y - half_crop),
        (center_x + half_crop, center_y - half_crop),
        (center_x - half_crop, center_y + half_crop),
        (center_x + half_crop, center_y + half_crop),
    ]

    min_x = min(x for x, y in corners)
    max_x = max(x for x, y in corners)
    min_y = min(y for x, y in corners)
    max_y = max(y for x, y in corners)

    if min_x < 0 or min_y < 0 or max_x > image.width or max_y > image.height:
        raise OutBoundaryException(
            "Cropped area exceeds image boundaries after rotation"
        )

    crop_box = (
        int(center_x - half_crop),
        int(center_y - half_crop),
        int(center_x + half_crop),
        int(center_y + half_crop),
    )
    cropped_image = image.crop(crop_box)

    return cropped_image


def generate_points(h: int, w: int):
    coordinate_h_list = [angle for angle in range(500, h - 500)]
    coordinate_w_list = [angle for angle in range(500, w - 500)]
    random.seed(time.time())
    random.shuffle(coordinate_h_list)
    random.shuffle(coordinate_w_list)
    center_x = random.choice(coordinate_h_list)
    center_y = random.choice(coordinate_w_list)
    center_point = (center_x, center_y)
    return center_point


def generate_training_ds_with_one_sat_img(sat_img, crop_size: int, num_of_images: int):
    start_points = set()
    h, w = sat_img.size
    while len(start_points) < num_of_images:
        print(f"Generate: {len(start_points)}")
        point = generate_points(h, w)
        start_points.add(point)
    start_point_list = list(start_points)

    # test_points = start_point_list[:5000]
    # count_train_points = math.ceil((num_of_images - 5000) * 0.8)
    # train_points = start_point_list[5000:count_train_points]
    # val_points = start_point_list[count_train_points:]

    base_path = f"~/Documents/rvl/pohsun/datasets/taipei_1"

    train_points = start_points
    print(f"train points: {len(train_points)}")
    output(crop_size, train_points, base_path, "train", sat_img)

    # print(f"val points: {len(val_points)}")
    # output(crop_size, val_points, base_path, "val", sat_img)
    #
    # print(f"test points: {len(test_points)}")
    # output(crop_size, test_points, base_path, "test", sat_img)

def output(crop_size: int, points: set, base_path: str, folder: str, sat_img):
    output_dir = Path(base_path).expanduser() / folder
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    for filename in output_dir.iterdir():
        if filename.is_file():
            filename.unlink()

    cropped_img_list = []
    rotated_img_list = []
    x_list = []
    y_list = []
    orientaion_list = []
    corner_list = []
    angles = [angle for angle in range(0, 360, 45)]
    # angles = [0] + angles
    sorted(angles)
    for point_idx, point in tqdm(enumerate(points), total=len(points), desc=f"Points {folder}"):
        center_point = point
        cropped_image = crop(sat_img, (center_point[0], center_point[1]), crop_size)
        output_path = output_dir / f"img_{point_idx}.jpg"
        cropped_image.save(output_path)
        for angle_id, angle in enumerate(angles):
            # print(f"angle: {angle}")
            try:
                # rotated_image = cropped_image.rotate(angle, expand=True)
                rotated_image, rotated_corners = rotate_and_crop(sat_img, center_point, angle, crop_size)
                rotated_output_path = output_dir / f"img_{point_idx}_{angle_id:01d}.jpg"
                rotated_image.save(rotated_output_path)
                cropped_img_list.append(output_path)
                rotated_img_list.append(rotated_output_path)
                corners = {
                    "location": [
                        {"top_left_u": rotated_corners[0][0], "top_left_v": rotated_corners[0][1]},
                        {"top_right_u": rotated_corners[1][0], "top_right_v": rotated_corners[1][1]},
                        {"bottom_left_u": rotated_corners[2][0], "bottom_left_v": rotated_corners[2][1]},
                        {"bottom_right_u": rotated_corners[3][0], "bottom_right_v": rotated_corners[3][1]}
                    ],
                    "orientation": angle
                }
                corner_list.append(corners)
            except OutBoundaryException as e:
                print(f"Exception: {e}")
                exit()

            x_list.append(center_point[0])
            y_list.append(center_point[1])
            orientaion_list.append(angle)

    df = pd.DataFrame(
        {
            "IMG_FILE": rotated_img_list,
            "TARGET_IMG_FILE": cropped_img_list,
            "LAT": x_list,
            "LON": y_list,
            "HEAD": orientaion_list,
            "CORNERS": corner_list
        }
    )

    df.to_csv(f"{base_path}/{folder}/taipei.csv", index=False)

    print(df.head(5))

def homo_output(crop_size: int, points: set, base_path: str, folder: str, sat_img):

    def create_and_clear(postfix: str):
        output_dir = Path(base_path).expanduser().joinpath(folder, postfix)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        for filename in output_dir.iterdir():
            if filename.is_file():
                filename.unlink()
        return output_dir

    src_img_output_dir = create_and_clear("original") 
    rotated_img_output_dir = create_and_clear("rotated")
    # label_dir = create_and_clear("label")

    angles = [random.randint(0, 359) for _ in range(len(points))]

    for point_idx, center_point in tqdm(enumerate(points), total=len(points), desc=f"Points {folder}"):
        x, y = center_point
        angle = angles[point_idx]
        patch_A = crop(sat_img, center_point, crop_size)
        patch_A_corners = [
            [x-crop_size, y-crop_size],
            [x+crop_size, y-crop_size],
            [x-crop_size, y+crop_size],
            [x+crop_size, y+crop_size]
        ]
        rotated_image, rotated_corners = rotate_and_crop(sat_img, center_point, angle, crop_size)
        # h = cv2.getPerspectiveTransform(patch_A_corners, rotated_corners)

        patch_A_path = src_img_output_dir.joinpath(f"{point_idx}.jpg")
        patch_A.save(str(patch_A_path))

        rotated_path = rotated_img_output_dir.joinpath(f"{point_idx}.jpg")
        rotated_image.save(str(rotated_path))

        data = {
            "src_corners": [
                {"top_left_u": patch_A_corners[0][0], "top_left_v": patch_A_corners[0][1]},
                {"top_right_u": patch_A_corners[1][0], "top_right_v": patch_A_corners[1][1]},
                {"bottom_left_u": patch_A_corners[2][0], "bottom_left_v": patch_A_corners[2][1]},
                {"bottom_right_u": patch_A_corners[3][0], "bottom_right_v": patch_A_corners[3][1]},
            ],
            "rotated_corners": [
                {"top_left_u": rotated_corners[0][0], "top_left_v": rotated_corners[0][1]},
                {"top_right_u": rotated_corners[1][0], "top_right_v": rotated_corners[1][1]},
                {"bottom_left_u": rotated_corners[2][0], "bottom_left_v": rotated_corners[2][1]},
                {"bottom_right_u": rotated_corners[3][0], "bottom_right_v": rotated_corners[3][1]},
            ],
            "center_x": x,
            "center_y": y,
            "angle": angle,
            "src_img": str(patch_A_path),
            "rotated_img": str(rotated_path)
        }
        label_file = label_dir.joinpath(f"{point_idx}_label.txt")
        with open(str(label_file), '+w') as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size", default=192, type=int, help="the size of cropped images"
    )
    parser.add_argument("--points", default=800, type=int, help="number of points")
    parser.add_argument("--input_sat_folder", type=str, help="satellite image folder")
    args = parser.parse_args()

    sat_img = Image.open("./datasets/satellites/202402_taipei.jpg")
    generate_training_ds_with_one_sat_img(sat_img, args.size, args.points)
