# Convert the data to dataloader formater.
import os

kaggle_dataset = "data/train/train/"

# Check root directory exists,
# If not, create it.
if not os.path.exists("root"):
    os.makedirs("root")

# Check kaggle dataset exists,
if not os.path.exists(kaggle_dataset):
    print("Kaggle dataset not found.")
    exit()

# Check if the labels.csv file exists, if it does, delete it.
if os.path.exists("root/labels.csv"):
    os.remove("root/labels.csv")

# Create a labels csv file.
print("Creating labels.csv file.")
with open("root/labels.csv", "w") as file:
    # file.write("image,class\n")
    classes = os.listdir(kaggle_dataset)
    for class_name in classes:
        image_files = os.listdir(os.path.join(kaggle_dataset, class_name))
        for image in image_files:
            class_name = class_name.replace("class", "")
            file.write(f"{image},{class_name}\n")

print("Creating uniform image dataset.")
# Create a uniform image dataset, named train


if not os.path.exists("root/train"):
    os.makedirs("root/train")

# Copy the images to the root directory.
for class_name in classes:
    image_files = os.listdir(os.path.join(kaggle_dataset, class_name))
    for image in image_files:
        os.system(f"cp {kaggle_dataset}/{class_name}/{image} root/train/{image}")