import os
import shutil

# Set up the directories
images_dir = "/Users/ac2349/GitHub/stability-computer-vision/data/images"
train_dir = "/Users/ac2349/GitHub/stability-computer-vision/data/traindir"
false_dir = os.path.join(train_dir, "0")
true_dir = os.path.join(train_dir, "1")

# Create the subdirectories if they don't exist
os.makedirs(false_dir, exist_ok=True)
os.makedirs(true_dir, exist_ok=True)

# Loop through the images and sort them based on the "True" or "False" in their filenames
for filename in os.listdir(images_dir):
    if "False" in filename:
        shutil.copy(os.path.join(images_dir, filename), false_dir)
    elif "True" in filename:
        shutil.copy(os.path.join(images_dir, filename), true_dir)
