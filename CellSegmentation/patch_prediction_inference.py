# Importing Libraries
import torch
from dataloader import FullSizeImageDataset, PatchesDataLoader
from model import build_unet
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

# Setting device to GPU
device = torch.device("cuda")

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((9*512,8*512))
])


# patching images:
def generate_patches(loader, size, stride, channels_img, channels_mask):
    for img, mask in loader:
        img = img.unfold(1, channels_img, channels_img).unfold(2, size, stride).unfold(3, size, stride)
        img = img.reshape(img.size(0) * img.size(2) * img.size(3), channels_img, size, size)
        mask = mask.unfold(1, 1, 1).unfold(2, size, stride).unfold(3, size, stride)
        mask = mask.reshape(mask.size(0) * mask.size(2) * mask.size(3), channels_mask, size, size)

    return img, mask

def plot_patch(img,mask, pred):
    img = img.squeeze(0).permute(1, 2, 0)
    mask = mask.squeeze(0).permute(1, 2, 0)
    mask = torch.cat([mask, mask, mask], dim=-1)
    pred = pred.squeeze(0).permute(1, 2, 0)
    pred = torch.cat([pred, pred, pred], dim=-1)
    line = (torch.ones((512, 10, 3)) * 128).to(device)
    img_grid = torch.cat([img, line, mask, line, pred], dim=1)

    cv2.imwrite("grid.jpg",img_grid.detach().cpu().numpy())


if __name__ == "__main__":
    train_dataset = FullSizeImageDataset("Colonic_crypt_dataset/train", "Colonic_crypt_dataset/train_mask",
                                         transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=1, pin_memory=True)
    img, mask = generate_patches(loader=train_loader,
                                 size=512,
                                 stride=512,
                                 channels_img=3,
                                 channels_mask=1)
    train_dataset = PatchesDataLoader(img.cpu(), mask.cpu())
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)
    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load("model_unet.h5"))

    # Prediction:
    model.eval()
    for img, mask in train_loader:
        img = img.to(device)
        mask = mask.to(device)
        pred = model(img)
        break

    # Plot
    plot_patch(img, mask, pred)