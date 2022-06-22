# Importing Libraries
import torch
import os
from dataloader import FullSizeImageDataset, PatchesDataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

# Setting device to GPU
device = torch.device("cuda")


def combine_patches(patches_dir):
    big_img = torch.zeros((9 * 512, 8 * 512))
    col_index = 0
    row_index = 0
    row_counter = 1
    col_counter = 1
    for image in os.listdir(patches_dir):
        image = torch.load(os.path.join(patches_dir, image))
        big_img[row_index:row_index + 512, col_index:col_index + 512] = image.squeeze(0).squeeze(0)
        if col_counter % 8 == 0:
            row_counter += 1
            row_index += 512
            col_counter = 1
            col_index = 0
        else:
            col_counter += 1
            col_index += 512

    return big_img


if __name__ == "__main__":
    full_img = combine_patches("patches")
    # save fig
    # plt.figure(figsize=(20, 20))
    # plt.imshow(torch.cat([full_img.unsqueeze(0), full_img.unsqueeze(0), full_img.unsqueeze(0)], dim=0).permute(1, 2, 0).detach().cpu().numpy())


    cv2.imwrite("prediction.jpg",torch.cat([full_img.unsqueeze(0), full_img.unsqueeze(0), full_img.unsqueeze(0)], dim=0).permute(1, 2, 0).detach().cpu().numpy())
