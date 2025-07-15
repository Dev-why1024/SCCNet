from torch.utils.data import Dataset
import os
from PIL import Image
import json
import torch
import numpy as np
from torchvision.transforms import transforms
from models.segment_anything import sam_model_registry, SamPredictor
import cv2
import matplotlib.pyplot as plt


def get_exam_patch_by_box_prompt(mask, image):
    binary_mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)

    if len(binary_mask.shape) == 3 and binary_mask.shape[0] == 1:
        binary_mask = np.squeeze(binary_mask, axis=0)

    if binary_mask.shape != np.array(image).shape[:2]:
        print(f"Error: Mask shape {binary_mask.shape} does not match image shape {np.array(image).shape[:2]}")
        return

    white_background = np.ones_like(np.array(image), dtype=np.uint8) * 255

    new_image = white_background * (1 - binary_mask[..., np.newaxis]) + np.array(image) * binary_mask[..., np.newaxis]

    new_image = Image.fromarray(new_image.astype(np.uint8))

    coords = np.column_stack(np.where(binary_mask > 0))
    if coords.size == 0:
        print("No object detected in the mask!")
        return

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped_image = new_image.crop((x_min, y_min, x_max, y_max))

    return cropped_image


def SamModel(SamModelPath, image, box, SamModelType="vit_h", device="cuda" if torch.cuda.is_available() else "cpu"):
    sam_checkpoint = SamModelPath

    model_type = SamModelType

    device = device

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    image = np.array(image)

    predictor.set_image(image)

    input_box = np.array(box)

    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    example_image = get_exam_patch_by_box_prompt(masks, image)

    return example_image
