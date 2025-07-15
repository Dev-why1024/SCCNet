import torch
import models.clip as clip
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from models.segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def find_most_similar_image(image_paths, text_prompt):
    model, preprocess = clip.load("ViT-B/32", device=device)

    images = [preprocess(img_path).unsqueeze(0).to(device) for img_path in image_paths]
    images_tensor = torch.cat(images, dim=0)

    text = clip.tokenize(text_prompt).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(images_tensor, text)
        logits = logits_per_image.cpu().numpy().flatten()

    max_logit_idx = logits.argmax()
    max_logit = logits[max_logit_idx]

    return {
        "index": max_logit_idx,
        "most_similar_image_path": image_paths[max_logit_idx],
        "logit_score": max_logit,
        "text_prompt": text_prompt
    }


def extract_and_save_objects_with_bbox(image, masks):
    cropped_images = []

    for i, ann in enumerate(masks):
        m = ann['segmentation']
        if m.ndim != 2:
            print(f"Mask {i} is not 2D, skipping...")
            continue

        mask = m.astype(bool)
        white_background = np.ones_like(image) * 255
        masked_object = np.where(mask[..., None], image, white_background)

        coords = np.column_stack(np.where(mask))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        cropped_result = masked_object[y_min:y_max + 1, x_min:x_max + 1]

        cropped_image = Image.fromarray(cropped_result.astype('uint8'))
        cropped_images.append(cropped_image)

    return cropped_images


if __name__ == '__main__':
    sam_checkpoint = "./checkpoints/bmnet+_pretrained/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    image_path = r"./DIR_OF_FSC147_DATASET/images_384_VarV2/216.jpg"
    pil_image = Image.open(image_path).convert("RGB")
    image = np.array(pil_image)

    masks = mask_generator.generate(image)

    print(f"Number of masks: {len(masks)}")
    print(f"Keys in the first mask: {masks[0].keys()}")

    cropped_images_list = extract_and_save_objects_with_bbox(image, masks)

    tile_images = cropped_images_list
    text_prompt = ["bread"]

    best_example_patch = find_most_similar_image(tile_images, text_prompt)

    tile_images[best_example_patch["index"]].show()

    # return tile_images[best_example_patch["index"]]
