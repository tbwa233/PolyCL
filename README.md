# PolyCL
This repository contains the implementation of PolyCL. Our publications for this project are listed below:

["PolyCL: Context-Aware Contrastive Learning for Image Segmentation,"](https://ieeexplore.ieee.org/abstract/document/10635698) by Aaron Moseley and Abdullah-Al-Zubaer Imran. In [IEEE ISBI](https://biomedicalimaging.org/2024/), 2024.

["Domain and Task-Focused Example Selection for Data-Efficient Contrastive Medical Image Segmentation,"](https://arxiv.org/abs/2505.19208) by Tyler Ward, Aaron Moseley, and Abdullah-Al-Zubaer Imran. In [arXiv:2505.19208](https://arxiv.org), 2025.

Our proposed model employs innovative example selection strategies to pre-train models on unlabeled data, significantly improving segmentation performance and generalizability while reducing reliance on labeled data​.

## Abstract
Segmentation is one of the most important tasks in the medical imaging pipeline as it influences a number of image-based decisions. To be effective, fully supervised segmentation approaches require large amounts of manually annotated training data. However, the pixel-level annotation process is expensive, time-consuming, and error-prone, hindering progress and making it challenging to perform effective segmentations. Therefore, models must learn efficiently from limited labeled data. Self-supervised learning (SSL), particularly contrastive learning via pre-training on unlabeled data and fine-tuning on limited annotations, can facilitate such limited labeled image segmentation. To this end, we propose a novel self-supervised contrastive learning framework for medical image segmentation, leveraging inherent relationships of different images, dubbed PolyCL. Without requiring any pixel-level annotations or unreasonable data augmentations, our PolyCL learns and transfers context-aware discriminant features useful for segmentation from an innovative surrogate, in a task-related manner. Experimental evaluations on three public computed tomography (CT) datasets demonstrate that PolyCL outperforms fully-supervised and self-supervised baselines in both low-data and cross-domain scenarios.

## Example Selection Strategies
<img src="https://github.com/tbwa233/PolyCL/blob/main/images/polyclteaser.png?raw=true" alt="Figure" style="width:50%;"/>
