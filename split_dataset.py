import os
import random
import shutil

original_dataset = "Dataset"
base_dir = "split_dataset"

train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")

classes = ["Crack", "Normal", "Patchwork", "Pothole"]

# =========================
# Create folders
# =========================

for folder in [train_dir, valid_dir, test_dir]:

    for cls in classes:

        os.makedirs(os.path.join(folder, cls), exist_ok=True)

# =========================
# Split Dataset
# =========================

for cls in classes:

    source_folder = os.path.join(original_dataset, cls)

    images = os.listdir(source_folder)

    random.shuffle(images)

    train_images = images[:75]
    valid_images = images[75:91]
    test_images = images[91:107]

    # Copy train images
    for image in train_images:

        src = os.path.join(source_folder, image)

        dst = os.path.join(train_dir, cls, image)

        shutil.copy(src, dst)

    # Copy validation images
    for image in valid_images:

        src = os.path.join(source_folder, image)

        dst = os.path.join(valid_dir, cls, image)

        shutil.copy(src, dst)

    # Copy test images
    for image in test_images:

        src = os.path.join(source_folder, image)

        dst = os.path.join(test_dir, cls, image)

        shutil.copy(src, dst)

print("Dataset split completed successfully!")