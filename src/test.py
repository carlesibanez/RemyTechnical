import torch
from utils.PansEggsDataset import PansEggsDataset 
from models.unet import UNet
from tqdm import tqdm
import numpy as np

import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default='../dataset/test')
    parser.add_argument("--model-dir", type=str, default='checkpoints/unet.pth')
    FLAGS = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    # Create dataset
    dataset = PansEggsDataset(folder_path=FLAGS.data_dir, mode='test', new_size=(400, 400))
    print('Dataset size: ', len(dataset))

    # Create dataloaders
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Create model
    model = UNet(in_channels=3, classes=3).to(device)
    print(model)

    # Load model
    checkpoint       = torch.load(FLAGS.model_dir, map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    print(f'Model loaded successfully.')

    # Test the model
    model.eval()
    for batch in tqdm(test_loader):
        images, original_size, img_name = batch

        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        mask = predicted.cpu().numpy()[0]
        mask = mask.astype(np.uint8)
        import pdb; pdb.set_trace()
        break
    