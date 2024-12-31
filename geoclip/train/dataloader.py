import os
import torch
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from os.path import exists
from PIL import Image as im
from torchvision import transforms
from torch.utils.data import Dataset


def img_train_transform():
    train_transform_list = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.CenterCrop(336),
        # transforms.CenterCrop(192),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform_list

def img_rotation_transform():
    return transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.CenterCrop(224),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def img_val_transform():
    val_transform_list = transforms.Compose([
            # transforms.CenterCrop(224),
            transforms.CenterCrop(336),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return val_transform_list    

def img_test_transform():
    test_transform_list = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ])
    return test_transform_list    


class GeoDataLoader(Dataset):
    """
    DataLoader for image-gps datasets.
    
    The expected CSV file with the dataset information should have columns:
    - 'IMG_FILE' for the image filename,
    - 'LAT' for latitude, and
    - 'LON' for longitude.
    
    Attributes:
        dataset_file (str): CSV file path containing image names and GPS coordinates.
        dataset_folder (str): Base folder where images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
        rotation_transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, dataset_file, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.images, self.coordinates, self.orientations = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f"Error reading {dataset_file}: {e}")

        images = []
        coordinates = []
        orientations = []

        for _, row in tqdm(dataset_info.iterrows(), desc="Loading image paths and coordinates"):
            filename = os.path.join(self.dataset_folder, row['IMG_FILE'])
            if exists(filename):
                images.append(filename)
                latitude = float(row['LAT'])
                longitude = float(row['LON'])
                head = float(row['HEAD']) # orientation
                coordinates.append([latitude, longitude])
                orientations.append(head)
            else:
                raise ValueError(f"{filename} not found")

        return images, coordinates, orientations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        coordinate = self.coordinates[idx]
        theta = self.orientations[idx]

        image = im.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # thetas = ((thetas+180)%360)-180
        # thetas = torch.tensor(thetas, dtype=torch.float32)
        # thetas = torch.tensor(thetas, dtype=torch.float32)

        # coordinates = torch.tensor(gps, dtype=torch.float32)
        # poses = torch.cat((coordinates, thetas), dim=1)

        return image, torch.tensor(coordinate, dtype=torch.float32), torch.tensor(theta)


class TaipeiDataLoader(Dataset):
    """
    DataLoader for image datasets.
    
    The expected CSV file with the dataset information should have columns:
    - 'ROT_IMG_FILE' for the rotated image filename,
    - 'REF_IMG_FILE' for the reference image filename
    - 'LAT' for latitude, and
    - 'LON' for longitude
    - 'HEAD' for heading angle.
    
    Attributes:
        dataset_file (str): CSV file path containing image names and GPS coordinates.
        dataset_folder (str): Base folder where images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
        rotation_transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, dataset_file, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.ref_imgs, self.rot_imgs, self.coordinates, self.orientations = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f"Error reading {dataset_file}: {e}")

        ref_imgs = []
        rot_imgs = []
        coordinates = []
        orientations = []

        for _, row in tqdm(dataset_info.iterrows(), desc="Loading image paths and coordinates"):
            ref_img = os.path.join(self.dataset_folder, row['REF_IMG']) # reference image which is not rotated
            rot_img = os.path.join(self.dataset_folder, row['IMG_FILE']) # query image which is rotated

            if exists(ref_img) and exists(rot_img):
                ref_imgs.append(ref_img)
                rot_imgs.append(rot_img)
                latitude = float(row['LAT'])
                longitude = float(row['LON'])
                head = float(row['HEAD']) # orientation
                coordinates.append([latitude, longitude])
                orientations.append(head)

        return ref_imgs, rot_imgs, coordinates, orientations

    def __len__(self):
        return len(self.ref_imgs)

    def __getitem__(self, idx):
        ref_img_path = self.ref_imgs[idx]
        rot_img_path = self.rot_imgs[idx]
        coordinate = self.coordinates[idx]
        theta = self.orientations[idx]

        ref_img = im.open(ref_img_path).convert('RGB')
        rot_img = im.open(rot_img_path).convert('RGB')
        
        # ref_transform = transforms.Compose([
        #     transforms.PILToTensor(),
        #     transforms.ConvertImageDtype(torch.float),
        # ])
        # ref_img = ref_transform(ref_img)
        if self.transform:
            ref_img = self.transform(ref_img)
            rot_img = self.transform(rot_img)

        # thetas = ((thetas+180)%360)-180
        # thetas = torch.tensor(thetas, dtype=torch.float32)
        # thetas = torch.tensor(thetas, dtype=torch.float32)
        rad = np.deg2rad(theta)
        sin_theta = np.sin(rad)
        cos_theta = np.cos(rad)
        heading = torch.tensor([sin_theta, cos_theta], dtype=torch.float32)

        return ref_img, rot_img, torch.tensor(coordinate, dtype=torch.float32), heading


class ImgGalleryDataloader(Dataset):
    def __init__(self, h5_path, data_key='images', target_key='labels'):
        super().__init__()
        self.h5_path = h5_path
        self.data_key = data_key
        self.target_key = target_key
        
        # Get dataset length from h5 file
        with h5py.File(self.h5_path, 'r') as h5_file:
            self.length = len(h5_file[self.data_key])
    
    def __getitem__(self, index):
        # Load data and target from HDF5
        target = None
        with h5py.File(self.h5_path, 'r') as h5_file:
            data = torch.from_numpy(h5_file[self.data_key][index]).float()

            if self.target_key is not None:
                target = torch.from_numpy(h5_file[self.target_key][index])
        return data, target
    
    def __len__(self):
        return self.length
