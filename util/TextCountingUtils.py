import shutil

import torch
import models.clip as clip
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from models.segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def find_most_similar_image(image_paths, text_prompt, device):
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


def find_top3_similar_images(image_paths, text_prompt, device):
    model, preprocess = clip.load("ViT-B/32", device=device)

    images = [preprocess(img_path).unsqueeze(0).to(device) for img_path in image_paths]
    images_tensor = torch.cat(images, dim=0)

    text = clip.tokenize(text_prompt).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(images_tensor, text)
        logits = logits_per_image.cpu().numpy().flatten()

    top3_indices = logits.argsort()[-3:][::-1]

    results = []
    for idx in top3_indices:
        results.append({
            "index": idx,
            "most_similar_image_path": image_paths[idx],
            "logit_score": logits[idx],
            "text_prompt": text_prompt
        })

    return results


# def find_top3_similar_images(image_paths, text_prompt, device):
#
#
#     model, preprocess = clip.load("ViT-B/32", device=device)
#
#
#     images = [preprocess(img_path).unsqueeze(0).to(device) for img_path in image_paths]
#     images_tensor = torch.cat(images, dim=0)
#
#
#     text = clip.tokenize(text_prompt).to(device)
#
#
#     with torch.no_grad():
#         logits_per_image, _ = model(images_tensor, text)
#         logits = logits_per_image.cpu().numpy().flatten()
#
#
#     top8_indices = logits.argsort()[-8:][::-1]
#
#
#     results = []
#     for idx in top8_indices:
#         results.append({
#             "index": idx,
#             "most_similar_image_path": image_paths[idx],
#             "logit_score": logits[idx],
#             "text_prompt": text_prompt
#         })
#
#     return results


def extract_and_list_objects_with_bbox(image, masks):
    cropped_images = []

    boxes = []

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

        boxes.append([x_min, y_min, x_max, y_max])

    return cropped_images, boxes


def get_image_class(image_name, annotation_file="./DIR_OF_FSC147_DATASET/ImageClasses_FSC147.txt"):
    try:
        with open(annotation_file, "r") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    img_name, class_label = parts
                    if img_name == image_name:
                        return class_label
    except FileNotFoundError:
        print(f"annotation file {annotation_file} not found.")
    except Exception as e:
        print(f"The error occurred while reading the annotation file: {e}")

    return None


def save_top3_images_and_scores(best_example_patches):
    with open("top3_scores.txt", "w", encoding="utf-8") as score_file:
        for rank, patch in enumerate(best_example_patches, start=1):
            img = patch["most_similar_image_path"]
            score = patch["logit_score"]
            dest_filename = f"top_similar_{rank}.jpg"
            img.save(dest_filename)
            score_file.write(f"{dest_filename}: {score}\n")


def SamClipModel(SamModelPath, image, text_prompt
                 , device="cuda" if torch.cuda.is_available() else "cpu", SamModelType="vit_h"):
    sam_checkpoint = SamModelPath
    model_type = SamModelType

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    image = np.array(image)

    masks = mask_generator.generate(image)

    cropped_images_list, boxes = extract_and_list_objects_with_bbox(image, masks)

    tile_images = cropped_images_list

    best_example_patches = find_top3_similar_images(tile_images, text_prompt, device)

    save_top3_images_and_scores(best_example_patches)

    return (
        [tile_images[patch["index"]] for patch in best_example_patches],
        [boxes[patch["index"]] for patch in best_example_patches]
    )

# def SamClipModel(SamModelPath, image, text_prompt
#                  , device="cuda" if torch.cuda.is_available() else "cpu", SamModelType="vit_h"):
#     sam_checkpoint = SamModelPath
#     model_type = SamModelType
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     mask_generator = SamAutomaticMaskGenerator(sam)
#     image = np.array(image)
#     masks = mask_generator.generate(image)
#     cropped_images_list, boxes = extract_and_list_objects_with_bbox(image, masks)
#     tile_images = cropped_images_list
#     best_example_patches = find_top3_similar_images(tile_images, text_prompt, device)
#     return (
#         [tile_images[patch["index"]] for patch in best_example_patches],
#         [boxes[patch["index"]] for patch in best_example_patches]
#     )
