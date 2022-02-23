import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pickle

# train_df_img_path_list = []
# train_df_img_label = []
# train_df = pd.DataFrame(columns=["img_path","label"])
# for idx, i in enumerate(os.listdir("Image Classification Data/data/train")):
#     for j in enumerate(os.listdir(f"Image Classification Data/data/train/{i}")):
#         train_df_img_path_list.append(f"Image Classification Data/data/train/{i}/{j[1]}")
#         train_df_img_label.append(idx)
# train_df["img_path"] = train_df_img_path_list
# train_df["label"] = train_df_img_label
# train_df.to_csv (r'train_csv.csv', index = False, header=True)

# test_df_img_path_list = []
# test_df_img_label = []
# test_df = pd.DataFrame(columns=["img_path","label"])
# for idx, i in enumerate(os.listdir("Image Classification Data/data/test")):
#     for j in enumerate(os.listdir(f"Image Classification Data/data/test/{i}")):
#         test_df_img_path_list.append(f"Image Classification Data/data/test/{i}/{j[1]}")
#         test_df_img_label.append(idx)
# test_df["img_path"] = test_df_img_path_list
# test_df["label"] = test_df_img_label
# test_df.to_csv (r'test_csv.csv', index = False, header=True)

class CustomDataloader(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img = Image.open(self.dataframe["img_path"][index]).convert("RGB")
        y_label = self.dataframe["label"][index]

        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)



class CNN(nn.Module):
    def __init__(self, output_dim=12):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels = 4,out_channels = 8,kernel_size = 3,padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels = 8,out_channels = 16,kernel_size = 3,padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,output_dim)
        )

    def forward(self, x):
        output = self.cnn_model(x)
        return output

def test_accuracy_check(loader, model):
    num_correct = 0  # keep track of correctly classified Images
    num_samples = 0  # keep track of batches checked
    model.eval()  # set model in evaluation mode for test accuracy calculation

    with torch.no_grad():  # no gradient calculations required during testing
        for imgs, labels in loader:
            imgs = imgs.to(device)  # shift images to GPU
            labels = labels.to(device)  # shift labels to GPU
            scores = model(imgs)  # predictions vector containing probability of each digit
            predictions = torch.tensor([torch.argmax(i).item() for i in scores]).to(device)  # extract digit index with the highest probability
            num_correct+= (predictions == labels).sum()  # calculating correctly classified images from the batch
            num_samples+= predictions.size(0)  # adding the batch size to total samples counter
    acc = float(num_correct)/float(num_samples)*100  # calculating accuracy
    print(f"Got {num_correct} / {num_samples} with test accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    model.train()  # put the model back in train for further epochs
    return acc


def train(batch_size):
    model_cnn.train()  # setting model to train mode
    epoch_lost_list = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(num_epochs):  # itereate through entire dataset in one epoch
        epoch_loss = 0
        num_correct = 0  # counter to keep track of the total correctly classified images from entire dataset per epoch
        num_samples = 0  # counter to keep track of the datapoints iterated in the entire dataset per epoch
        print("epoch:", epoch + 1)  # prints epoch number
        for imgs, labels in train_dataloader:  # trainloader returns a tuple -> (batch of images, corresponding vector of labels)
            # Feedforward Section
            imgs = imgs.to(device)  # shift images to GPU for faster training
            labels = labels.to(device)  # shift labels to GPU for faster training
            outputs = model_cnn(imgs)  # output of feedforward neural network before softmax layer
            # Backpropagation Section
            loss = criterion(outputs, labels)  # calculate the softmax output and loss per batch of the images
            optimizer.zero_grad()  # set the optimizer matrix to zero before calculating the gradients for every batch
            loss.backward()  # calculate the gradients through differentiation (dL/dW)
            epoch_loss += loss.item()
            optimizer.step()  # updation of weights (w = w - dL/dW)
            # Prediction Section
            predictions = torch.tensor([torch.argmax(i).item() for i in outputs]).to(
                device)  # using trained weights to predict the output.
            num_correct += (
                        predictions == labels).sum()  # if predictions vector matches labels vector, we increment num_correct by the number of correct predictions
            num_samples += predictions.size(0)  # increment the number of samples by batchsize
        epoch_lost_list.append(epoch_loss / (len(train_dataset) // batch_size))
        train_accuracies.append(float(num_correct) / float(num_samples) * 100)
        print(
            f"Got {num_correct} / {num_samples} with train accuracy {float(num_correct) / float(num_samples) * 100:.2f}\n")  # print training accuracy per epoch
        print("Train Loss Epoch: ", epoch_loss / (len(train_dataset) // batch_size))
        # Calculate test accuracy every epoch
        test_acc = test_accuracy_check(test_dataloader, model_cnn)
        test_accuracies.append(test_acc)
        torch.save(model_cnn, f"trained_classification_model_{epoch + 1}.pt")

    file_name_1 = "train_loss.pkl"
    file_name_2 = "train_acc.pkl"
    file_name_3 = "test_acc.pkl"
    open_file_1 = open(file_name_1, "wb")
    open_file_2 = open(file_name_2, "wb")
    open_file_3 = open(file_name_3, "wb")
    pickle.dump(epoch_lost_list, open_file_1)
    pickle.dump(train_accuracies, open_file_2)
    pickle.dump(test_accuracies, open_file_3)
    open_file_1.close()
    open_file_2.close()
    open_file_3.close()



if __name__ == '__main__':
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv("train_csv.csv")
    test_df = pd.read_csv("test_csv.csv")

    transform = transforms.Compose([
        transforms.Resize((2000, 2000)),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataloader(dataframe=train_df, transform=transform)
    test_dataset = CustomDataloader(dataframe=test_df, transform=transform)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, pin_memory=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=16, pin_memory=False, num_workers=4)

    num_epochs = 20
    learning_rate = 0.001

    model_cnn = CNN().to(device)  # model will run on GPU
    criterion = nn.CrossEntropyLoss()  # function callout for softmax output and loss calculation
    optimizer = torch.optim.Adam(model_cnn.parameters(), lr=learning_rate)  # Optimizer set to Adam

    train(8)