from PIL import Image

from main import ResNet18
import torch, os
import os
from PIL import Image
import torch
from torchvision import transforms
# Submission of code,
from tqdm import tqdm


device = torch.device("mps")

'''

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import idx2numpy
from torchvision.io import read_image
from torchvision.transforms import v2
from PIL import Image
import pandas as pd
import os
import torch
from tqdm import tqdm
import cv2
IMAGE_DIMS = 224

device = torch.device("mps")


# 128x128 images
transforms = v2.Compose([
    v2.Resize(size=(IMAGE_DIMS, IMAGE_DIMS)),
    # convert to rgb from greyscale.
    #v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomPerspective(distortion_scale=0.6, p=0.4),
    # v2.GaussianBlur(kernel_size=(5, 11), sigma=(0.1, 0.2)),
    v2.RandomRotation(degrees=(-15, 15)),
    v2.RandomAffine(degrees=(-15, 15)),
    #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

 # batch size, width, height, channels.  Bx128x128x3


class CustomImageDataset(Dataset):
    """
        This class must inherit from the torch.utils.data.Dataset class.
        And contina functions __init__, __len__, and __getitem__.
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        """Get the image and label at the index idx."""
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        Image.open(img_path).convert("RGB").save(img_path)
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# test_data = CustomImageDataset("data/", "mnist_dataset/t10k-images.idx3-ubyte" )  #, transform=test_transform)
# print((test_data[0])[0].shape, "label value", test_data[0][1]) # Getting image from dataset.
train_data = CustomImageDataset("./root/labels.csv", "./root/train/", transform=transforms)

# Create a DataLoader, so we can iterate through the dataset in batches.
#train_loader = DataLoader(train_data, batch_size=64, shuffle=True, )

# Testing the dataloader.
# for i, (images, labels) in enumerate(train_loader):
#     print(i, images.shape, labels.shape)


train_size = int(0.8 * len(train_data))
test_size = len(train_data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

# Create DataLoader for train and test sets
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# RESNet has many different configurations such as 18, 34, 50, 101, 152.
import gc
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
import torchvision.datasets as datasets
from torchvision import transforms
from torch import nn
import random

mps_device = torch.device("mps")
x = torch.ones(1, device=mps_device)
# Path: model.py

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


loss = nn.CrossEntropyLoss()


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.01):  # conv_arch:
        super(ResNet18, self).__init__()
        self.out_classes = num_classes
        self.drop = nn.Dropout(p=dropout_rate)
        # Define the ResNet34 architecture.
        self.conv0 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=(7, 7),
            stride=2, padding=3, bias=False,
        )
        self.bn0 = nn.BatchNorm2d(num_features=64)
        self.relu0 = nn.ReLU()
        self.maxpool0 = nn.MaxPool2d(
            kernel_size=(3, 3), stride=2, padding=1,
        )

        # First part of first conv block
        self.conv1_1 = nn.Conv2d(
            in_channels=64, out_channels=64,
            kernel_size=(3, 3),
            stride=1, padding=1, bias=False,
        )
        self.bn1_1 = nn.BatchNorm2d(num_features=64)
        self.relu1_1 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(
            in_channels=64, out_channels=64,
            kernel_size=(3, 3),
            stride=1, padding=1, bias=False,
        )
        self.bn1_2 = nn.BatchNorm2d(num_features=64)
        self.relu1_2 = nn.ReLU()
        # Second part of first conv block
        self.conv1_3 = nn.Conv2d(
            in_channels=64, out_channels=64,
            kernel_size=(3, 3),
            stride=1, padding=1, bias=False,
        )
        self.bn1_3 = nn.BatchNorm2d(num_features=64)
        self.relu1_3 = nn.ReLU()

        self.conv1_4 = nn.Conv2d(
            in_channels=64, out_channels=64,
            kernel_size=(3, 3),
            stride=1, padding=1, bias=False,
        )
        self.bn1_4 = nn.BatchNorm2d(num_features=64)
        self.relu1_4 = nn.ReLU()

        # First part of second conv block, this layer changes features to 128
        self.conv2_1 = nn.Conv2d(
            in_channels=64, out_channels=128,
            kernel_size=(3, 3),
            stride=2, padding=1, bias=False,
        )
        self.bn2_1 = nn.BatchNorm2d(num_features=128)
        self.relu2_1 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(
            in_channels=128, out_channels=128,
            kernel_size=(3, 3),
            stride=1, padding=1, bias=False,
        )
        self.bn2_2 = nn.BatchNorm2d(num_features=128)
        self.relu2_2 = nn.ReLU()

        self.dim_match_conv1 = nn.Conv2d(
            in_channels=64, out_channels=128,
            kernel_size=(1, 1),
            stride=2, bias=False,
        )
        self.dim_match_bn_1 = nn.BatchNorm2d(128)
        self.dim_match_relu_1 = nn.ReLU()
        # Second part of second conv block
        self.conv2_3 = nn.Conv2d(
            in_channels=128, out_channels=128,
            kernel_size=(3, 3),
            stride=1, padding=1, bias=False,
        )
        self.bn2_3 = nn.BatchNorm2d(num_features=128)
        self.relu2_3 = nn.ReLU()

        self.conv2_4 = nn.Conv2d(
            in_channels=128, out_channels=128,
            kernel_size=(3, 3),
            stride=1, padding=1, bias=False,
        )
        self.bn2_4 = nn.BatchNorm2d(num_features=128)
        self.relu2_4 = nn.ReLU()

        self.relu_after_add_1 = nn.ReLU()

        # First part of third conv block
        self.conv3_1 = nn.Conv2d(
            in_channels=128, out_channels=256,
            kernel_size=(3, 3),
            stride=2, padding=1, bias=False,
        )
        self.bn3_1 = nn.BatchNorm2d(num_features=256)
        self.relu3_1 = nn.ReLU()

        self.conv3_2 = nn.Conv2d(
            in_channels=256, out_channels=256,
            kernel_size=(3, 3),
            stride=1, padding=1, bias=False,
        )
        self.bn3_2 = nn.BatchNorm2d(num_features=256)
        self.relu3_2 = nn.ReLU()

        self.dim_match_conv2 = nn.Conv2d(
            in_channels=128, out_channels=256,
            kernel_size=(1, 1),
            stride=2, bias=False,
        )
        self.dim_match_bn_2 = nn.BatchNorm2d(256)
        self.dim_match_relu_2 = nn.ReLU()
        # Second part of third conv block
        self.conv3_3 = nn.Conv2d(
            in_channels=256, out_channels=256,
            kernel_size=(3, 3),
            stride=1, padding=1, bias=False,
        )
        self.bn3_3 = nn.BatchNorm2d(num_features=256)
        self.relu3_3 = nn.ReLU()

        self.conv3_4 = nn.Conv2d(
            in_channels=256, out_channels=256,
            kernel_size=(3, 3),
            stride=1, padding=1, bias=False,
        )
        self.bn3_4 = nn.BatchNorm2d(num_features=256)
        self.relu3_4 = nn.ReLU()

        self.relu_after_add_2 = nn.ReLU()

        # First part of fourth conv block
        self.conv4_1 = nn.Conv2d(
            in_channels=256, out_channels=512,
            kernel_size=(3, 3),
            stride=2, padding=1, bias=False,
        )
        self.bn4_1 = nn.BatchNorm2d(num_features=512)
        self.relu4_1 = nn.ReLU()

        self.conv4_2 = nn.Conv2d(
            in_channels=512, out_channels=512,
            kernel_size=(3, 3),
            stride=1, padding=1, bias=False,
        )
        self.bn4_2 = nn.BatchNorm2d(num_features=512)
        self.relu4_2 = nn.ReLU()

        self.dim_match_conv3 = nn.Conv2d(
            in_channels=256, out_channels=512,
            kernel_size=(1, 1),
            stride=2, bias=False,
        )
        self.dim_match_bn_3 = nn.BatchNorm2d(512)
        self.dim_match_relu_3 = nn.ReLU()
        # Second part of fourth conv block
        self.conv4_3 = nn.Conv2d(
            in_channels=512, out_channels=512,
            kernel_size=(3, 3),
            stride=1, padding=1, bias=False,
        )
        self.bn4_3 = nn.BatchNorm2d(num_features=512)
        self.relu4_3 = nn.ReLU()

        self.conv4_4 = nn.Conv2d(
            in_channels=512, out_channels=512,
            kernel_size=(3, 3),
            stride=1, padding=1, bias=False,
        )
        self.bn4_4 = nn.BatchNorm2d(num_features=512)
        self.relu4_4 = nn.ReLU()

        self.relu_after_add_3 = nn.ReLU()

        # Avg pool out put and pass through a FC connected layer which has 10 outputs for CIFAR10
        # (Change to 1000 for ImageNet)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        # Final FC layer - 10 for CIFAR 1000 for ImageNette
        self.fc1 = nn.Linear(
            in_features=512, out_features=10
            # in_features=512, out_features=1000
        )



    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool0(x)
        res_x = x.detach().clone()  # The x to be passed forward to the +2 layer
        # print(f'dim after init conv: {x.shape}')

        # First residual block of first conv block
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = x + res_x
        # res_x = x.detach().clone()  This might be the wrong approach for making res_x
        res_x = x
        # print(f'dim after first res block in first conv block {x.shape}')

        # Second residual block of first conv block
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = x + res_x
        # res_x = x.detach().clone()
        res_x = x
        # print(f'dim after second res block in first conv block {x.shape}')

        # First residual block of second conv block
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        # Must perform a (1x1) conv on res_x to make the dimensions match
        res_x = self.dim_match_conv1(res_x)
        res_x = self.dim_match_bn_1(res_x)
        # print(f'dim res_x after 1x1 conv: {res_x.shape}')
        x = x + res_x
        x = self.dim_match_relu_1(x)
        # res_x = x.detach().clone()
        res_x = x
        # print(f'dim after first res block in second conv block {x.shape}')

        # Second residual block of second conv block
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)
        x = x + res_x
        x = self.relu_after_add_1(x)
        # res_x = x.detach().clone()
        res_x = x
        # print(f'dim after second res block in second conv block {x.shape}')

        # First residual block of third conv block
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu3_2(x)

        # Must perform a (1x1) conv on res_x to make the dimensions match
        res_x = self.dim_match_conv2(res_x)
        res_x = self.dim_match_bn_2(res_x)
        # print(f'dim res_x after 1x1 conv: {res_x.shape}')
        x = x + res_x
        x = self.dim_match_relu_2(x)
        # res_x = x.detach().clone()
        res_x = x
        # print(f'dim after first res block in third conv block {x.shape}')

        # Second residual block of third conv block
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.relu3_3(x)
        x = self.conv3_4(x)
        x = self.bn3_4(x)
        x = self.relu3_4(x)
        x = x + res_x
        x = self.relu_after_add_2(x)
        # res_x = x.detach().clone()
        res_x = x
        # print(f'dim after second res block in third conv block {x.shape}')

        # First residual block of fourth conv block
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu4_2(x)

        # Must perform a (1x1) conv on res_x to make the dimensions match
        res_x = self.dim_match_conv3(res_x)
        res_x = self.dim_match_bn_3(res_x)
        # print(f'dim res_x after 1x1 conv: {res_x.shape}')
        x = x + res_x
        x = self.dim_match_relu_3(x)
        # res_x = x.detach().clone()
        res_x = x
        # print(f'dim after first res block in fourth conv block {x.shape}')

        # Second residual block of third conv block
        x = self.conv4_3(x)
        x = self.bn4_3(x)
        x = self.relu4_3(x)
        x = self.conv4_4(x)
        x = self.bn4_4(x)
        x = self.relu4_4(x)
        x = x + res_x
        x = self.relu_after_add_3(x)
        # res_x = x.detach().clone()
        # print(f'dim after second res block in fourth conv block {x.shape}')

        x = self.global_avg_pool(x)
        # print(f'dim after global avg pool: {x.shape}')
        # Flatten after pooling to make output into a vector ready for FC layer
        x = torch.flatten(x, start_dim=1)
        # print(f'dim after flatten: {x.shape}')



        # Final fc layer
        x = self.drop(x)
        x = self.fc1(x)
        # print(f'Final output dim: {x.shape}')

        return x







if __name__ == "__main__":
    # Test the model.

    EPOCH = 35
    model = ResNet18()
    #model.apply(init_weights)
    model.load_state_dict(torch.load('resnet18_62.pth'))
    model.to(device)
    print(model)
    print("Model has been tested and is working correctly.")
    # Running the model with test data.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    if os.path.exists("best_model.txt"):
        with open("best_model.txt", "r") as file:
            best_accuracy = float(file.read())
    else: best_accuracy = 0

    for epoch in tqdm(range(EPOCH)):
        print(f"Running Epoch {epoch}")
        model.train()  # Set model to training mode
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}] training loss: {train_loss:.3f}')


        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data in test_loader:  # Assuming test_loader is used as a validation loader
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(test_loader)
        val_accuracy = 100 * val_correct / val_total
        print(f'Epoch [{epoch + 1}] validation loss: {val_loss:.3f}, accuracy: {val_accuracy:.2f}%')
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "resnet18.pth")
            with open("best_model.txt", "w") as file:
                file.write(f"{best_accuracy}")
            print("parameters saved")

        # Update the LR scheduler with validation loss
        scheduler.step(val_loss)
        print(f'LR: {scheduler.get_last_lr()}')

'''

TEST_DIR = 'data/test/test/'

model = ResNet18()
model.load_state_dict(torch.load('resnet18.pth'))
model.eval()

# Define a transformation to preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the input size expected by the model
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

images = os.listdir(TEST_DIR)
print(len(images))

with open("submission.csv", "w") as file:
    for image in tqdm(images):
        img_path = os.path.join(TEST_DIR, image)
        Image.open(img_path).convert("RGB").save(img_path)
        img = Image.open(img_path)
        img = transform(img)  # Apply the defined transformation
        img = img.unsqueeze(0)  # Add a batch dimension

        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            image_name = image.split(".")[0]
            file.write(f"{image_name},{predicted.item()}\n")