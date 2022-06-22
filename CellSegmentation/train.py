import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
from dataloader import FullSizeImageDataset, PatchesDataLoader
from model import build_unet
from loss import DiceBCELoss

# setting device to GPU:
device = torch.device("cuda")

# Setting transformations for image data loaders:
transform = transforms.Compose([
    transforms.Resize((9 * 512, 8 * 512))
])


# patching images:
def generate_patches(loader, size, stride, channels_img, channels_mask):
    for img, mask in loader:
        img = img.unfold(1, channels_img, channels_img).unfold(2, size, stride).unfold(3, size, stride)
        img = img.reshape(img.size(0) * img.size(2) * img.size(3), channels_img, size, size)
        mask = mask.unfold(1, 1, 1).unfold(2, size, stride).unfold(3, size, stride)
        mask = mask.reshape(mask.size(0) * mask.size(2) * mask.size(3), channels_mask, size, size)

    return img, mask


# Training Loop:
def train(args, model, device, train_loader, optimizer, epoch, loss_fn):
    train_loss = 0
    model.train()  # setting model to train mode
    for imgs, labels in train_loader:  # train loader returns a tuple -> (batch of images, corresponding vector of labels)
        # Feed-forward Section
        imgs = imgs.to(device)  # shift images to GPU for faster training
        labels = labels.to(device)  # shift labels to GPU for faster training
        outputs = model(imgs)  # output of feed-forward neural network before softmax layer
        # Back-propagation Section
        loss = loss_fn(outputs, labels)  # calculate the softmax output and loss per batch of the images
        train_loss += loss.item()
        optimizer.zero_grad()  # set the gradients' matrix to zero before calculating the gradients for every batch
        loss.backward()  # calculate the gradients through differentiation (dL/dW)
        optimizer.step()  # update of weights (w = w - dL/dW)

    wandb.log({
        "Train Loss": train_loss / (len(train_loader.dataset) / imgs.size(0))})


def validation(args, model, device, valid_loader, epoch, loss_fn):
    valid_loss = 0
    model.eval()
    for imgs, labels in valid_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        valid_loss += loss.item()

    wandb.log({
        "Validation Loss": valid_loss / (len(valid_loader.dataset) / imgs.size(0))})

    return valid_loss / (len(valid_loader.dataset) / imgs.size(0))

# Set Hyper parameters and wandb dashboard setup:
# WandB – Initialize a new run
wandb.init(entity="ajinkya98", project="pytorch-cell-segmentation")
wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config  # Initialize config for train
config.batch_size = 4  # input batch size for training (default: 64)
# config.test_batch_size = 1000  # input batch size for testing (default: 1000)
config.epochs = 50  # number of epochs to train (default: 10)
config.lr = 0.0001  # learning rate (default: 0.01)
# config.momentum = 0.1          # SGD momentum (default: 0.5)
config.seed = 42  # random seed (default: 42)
config.log_interval = 10  # how many batches to wait before logging training status
config.valid_batch_size = 1

# Set random seeds and deterministic pytorch for reproducibility
# random.seed(config.seed)       # python random seed
torch.manual_seed(config.seed)  # pytorch random seed
# numpy.random.seed(config.seed) # numpy random seed
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Loading the datasets and data loaders
    train_dataset = FullSizeImageDataset("Colonic_crypt_dataset/train", "Colonic_crypt_dataset/train_mask",
                                         transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=1,
                              pin_memory=True)
    valid_dataset = FullSizeImageDataset("Colonic_crypt_dataset/test", "Colonic_crypt_dataset/test_mask",
                                         transform=transform)
    valid_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1,
                              pin_memory=True)
    img, mask = generate_patches(loader=train_loader,
                                 size=512,
                                 stride=512,
                                 channels_img=3,
                                 channels_mask=1)
    img_valid, mask_valid = generate_patches(loader=valid_loader,
                                             size=512,
                                             stride=512,
                                             channels_img=3,
                                             channels_mask=1)
    train_dataset = PatchesDataLoader(img, mask)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=1)
    valid_dataset = PatchesDataLoader(img_valid, mask_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True, pin_memory=True, num_workers=1)
    model = build_unet()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    wandb.watch(model, log="all")
    min_loss = 1000
    loss_fn = DiceBCELoss()
    for epoch in range(1, config.epochs + 1):
        train(config, model, device, train_loader, optimizer, epoch, loss_fn)
        loss = validation(config, model, device, valid_loader, epoch, loss_fn)
        # WandB – Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
        if loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), "model_unet.h5")
    #     wandb.save('model_unet.h5')


