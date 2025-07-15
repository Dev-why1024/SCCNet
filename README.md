# SAM-CLIP Counting Network

SAM-CLIP Counting Net: An arbitrary-shot framework for category-agnostic object counting in few-shot 
and zero-shot scenarios, leveraging CLIP and SAM for zero-shot generalization. It utilizes optimized modules, 
including BCM, EAFE, CSFF, and DACA, to enhance object localization and feature representation.

# Overview
Overview of the proposed method. The proposed framework supports both zero-shot and few-shot object counting tasks. In zero-shot scenarios, the
system leverages the SAM model for panoptic segmentation, followed by similarity matching using the CLIP model and text prompts to select image patches
with the highest similarity, which are then input into the counting model. In few-shot scenarios, the system relies on user-provided bounding box coordinates,
using the SAM model to automatically segment exemplar images, with the segmented results subsequently fed into the counting model for object estimation.
This method combines zero-shot and few-shot learning strategies, offering an efficient and flexible solution for object counting.

<img src="./figures/pipeline.png" alt="Pipeline" style="width: 100%;">

## Acknowledgment
This repo heavily based on [BMNet]([https://link-url-here.org](https://github.com/flyinglynx/Bilinear-Matching-Network)https://github.com/flyinglynx/Bilinear-Matching-Network). Thanks for the great work.