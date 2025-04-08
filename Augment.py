import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define augmentation pipeline (only flipping and rotations)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(1.0, 1.0), translate_percent=(0, 0), rotate=(-30, 30), p=0.5),
])

# Input directories
input_dir = "C:/Users/janga/cancer/Lung_and_Colon_Cancer"

# Filter out non-directory files (fix NotADirectoryError)
classes = [cls for cls in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, cls))]
total_classes = len(classes)

for class_idx, class_name in enumerate(classes, start=1):
    class_input_dir = os.path.join(input_dir, class_name)

    images = [img for img in os.listdir(class_input_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(images)

    for img_idx, img_name in enumerate(images, start=1):
        # Check if augmented version already exists
        aug_img_name = f"{os.path.splitext(img_name)[0]}_aug.jpg"
        aug_img_path = os.path.join(class_input_dir, aug_img_name)
        if os.path.exists(aug_img_path):
            continue  # Skip if augmentation already exists

        img_path = os.path.join(class_input_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation
        augmented = transform(image=image)
        image_aug = augmented['image']

        # Save augmented image in the same folder
        cv2.imwrite(aug_img_path, cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR))

        # Print progress
        print(f"[{class_idx}/{total_classes}] Processed {img_idx}/{total_images} images in class '{class_name}'")

print("Augmentation completed successfully!")
