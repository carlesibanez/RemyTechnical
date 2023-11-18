import os
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import numpy as np


class PansEggsDataset(Dataset):
    def __init__(self, folder_path, masks=True, new_size=(1000, 1000)):
        """
        Args:
            folder_path (string): Path to the folder containing images and masks
            masks (bool): If True, masks are loaded as well (False for testing)
            new_size (tuple): Size to which images and masks are resized
        """
        self.folder_path = folder_path
        self.imgs = sorted(os.listdir(os.path.join(folder_path, 'images')))
        if masks:
            self.masks = sorted(os.listdir(os.path.join(folder_path, 'masks')))
            assert len(self.imgs) == len(self.masks), 'Number of images and masks must be equal'
        self.new_size = new_size

        # Set transform (convert to tensor and pad to max height and width)
        self.transform = Compose([ToTensor(), SquarePad(), Resize(self.new_size, antialias=True)])
        
        # Get max height and width (Informative only)
        self.get_max_size()
        print('Max height: ', self.max_height)
        print('Max width: ', self.max_width)
        
    def get_max_size(self):
        self.max_height = 0
        self.max_width = 0

        for img_name in self.imgs:
            img = Image.open(os.path.join(self.folder_path, 'images', img_name))
            w, h = img.size
            self.max_width = max(self.max_width, w)
            self.max_height = max(self.max_height, h)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Retrive image
        img_name = os.path.join(self.folder_path, 'images', self.imgs[idx])
        image = Image.open(img_name)        
        image = self.transform(image)
        image = image / 255.0
        
        # Retrive mask
        if self.masks:
            mask_name = os.path.join(self.folder_path, 'masks', self.masks[idx])
            mask = Image.open(mask_name)
            mask = mask.convert('L')
            mask = self.transform(mask) * 255.0
            # Convert to 3 classes (0 -> background, 1 -> egg, 2 -> pan)
            mask[mask <= 64] = 0
            mask[torch.bitwise_and(mask > 64, mask <= 192)] = 1
            mask[mask > 192] = 2
            mask = mask[0,:,:].long() 
        else:
            mask = None

        return image, mask


class SquarePad:
	def __call__(self, image):
		h, w = image.shape[-2:]
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')


# Not in use
class NewPad:
    def __init__(self, height, width):
        self.height = height
        self.width = width
    def __call__(self, image):
        h, w = image.shape[-2:]
        h_padding = (self.width - w) / 2   # Horizontal padding
        v_padding = (self.height - h) / 2  # Vertical padding
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
        return F.pad(image, padding, 0, 'constant')
