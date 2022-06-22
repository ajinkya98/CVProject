from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch

# DataLoader for full sized images
class FullSizeImageDataset(Dataset):
    def __init__(self, folder_img, folder_mask, transform):
        self.path_img = sorted \
            (list(filter(None ,[os.path.join(folder_img ,i) if "tiff" in i else None for i in os.listdir(folder_img)])))
        self.path_mask = sorted(list
            (filter(None ,[os.path.join(folder_mask ,i) if "tiff" in i else None for i in os.listdir(folder_mask)])))
        self.transform = transform
    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.path_img[index], cv2.IMREAD_COLOR)
        image = image /255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.path_mask[index], cv2.IMREAD_GRAYSCALE)
        mask = mask /255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return self.transform(image), self.transform(mask)

    def __len__(self):
        return len(self.path_img)

# Dataloader for patches:
class PatchesDataLoader(Dataset):
  def __init__(self,patched_tensor_imgs, patched_tensor_masks):
    self.patched_tensor_imgs = patched_tensor_imgs
    self.patched_tensor_masks = patched_tensor_masks

  def __getitem__(self, index):
      return self.patched_tensor_imgs[index], self.patched_tensor_masks[index]

  def __len__(self):
    return self.patched_tensor_imgs.size(0)
