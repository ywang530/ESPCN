import os
from os.path import join
from os import listdir

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, Scale

class ImageDataset(Dataset):
    """ Custom Image Dataset """
    def __init__(self, root_dir, upscale_factor, input_transform=None, target_transform=None):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        
        self.image_dir = root_dir + '/UPSCALE_X' + str(upscale_factor) + '/data'
        self.target_dir = root_dir + '/UPSCALE_X' + str(upscale_factor) + '/target'
        self.image_filenames = [join(self.image_dir, x) for x in listdir(self.image_dir)]
        self.target_filenames = [join(self.target_dir, x) for x in listdir(self.target_dir)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_filenames)

    
    def __getitem__(self, idx):
        image = Image.open(self.image_filenames[idx]).convert(mode='YCbCr')
        image_Y, image_Cb, image_Cr = image.split()

        target = Image.open(self.target_filenames[idx]).convert(mode='YCbCr')
        target_Y, target_Cb, target_Cr = target.split()

        if self.input_transform:
            image_Y = self.input_transform(image_Y)

        if self.target_transform:
            target_Y = self.target_transform(target_Y)

        return image_Y, target_Y

    