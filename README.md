# WeedSAM

**Weakly supervised crop and weed segmentation from image-level labels**

WeedSAM is a precision agriculture proof of concept that turns plant-presence predictions into pixel-level maps. It tests whether automatically prompted Segment Anything 2 (SAM 2) and an Excess Green (ExG) vegetation filter improve Grad-CAM pseudo-masks on field images.

**Evaluation status.** Image-level classification on the held-out test split and the segmentation ablation on the validation split are complete. **Segmentation test evaluation is pending** for all four methods on 46 held-out images.

## Dataset

The experiment uses 300 RGB field images from CropAndWeed with corresponding pixel-level label maps. The original categories are remapped to three foreground targets. Maize combines source label IDs 1–6, Meadow-grass uses ID 48, and Creeping thistle uses ID 38. Class presence provides the image-level training labels for ResNet18. Pixel masks support validation threshold selection and segmentation scoring. [CropAndWeed dataset and documentation](https://github.com/cropandweed/cropandweed-dataset)

| Split | Images | Recording sessions | Role |
| --- | ---: | ---: | --- |
| Training | 210 | 21 | Fit the multi-label classifier |
| Validation | 44 | 5 | Select the checkpoint and thresholds, compare segmentation variants |
| Test | 46 | 5 | Evaluate image-level classification and, once completed, segmentation |

The saved splits are grouped by recording session, with no session shared across splits.

## Pipeline

1. **Image-level classification.** ResNet18 takes a 448 × 448 RGB input and predicts independent sigmoid probabilities for the three targets. The class thresholds used for segmentation are 0.10 for Maize, 0.10 for Meadow-grass, and 0.15 for Creeping thistle.
2. **Weak localization.** Grad-CAM produces one heatmap per predicted class from the final ResNet18 residual block. Heatmaps are resized to the original image dimensions and thresholded at 0.70. The strongest activation determines the label where classes overlap.
3. **Prompted mask refinement.** Each connected activation region of at least 50 pixels provides a positive point at the activation peak and a bounding box padded by 15% on each side. Pretrained SAM 2.1 Hiera Tiny returns candidate masks for each prompt. The highest-scoring candidate is retained, and overlaps are resolved using the product of the class probability and SAM 2 mask score. The resulting map assigns one semantic label per pixel. [SAM 2 implementation and checkpoints](https://github.com/facebookresearch/sam2)
4. **Vegetation filtering.** The normalized Excess Green index, `(2G − R − B) / (R + G + B + ε)`, retains predicted foreground pixels with ExG ≥ 0.05. The same filter is evaluated on both the Grad-CAM and SAM 2 masks.

## Classification test results

The saved image-level comparison covers the 46 held-out images. The tuned thresholds were used to gate Grad-CAM in the segmentation experiments.

| Classifier configuration | Macro F1 | Micro F1 |
| --- | ---: | ---: |
| Baseline | **0.678** | **0.636** |
| Validation-tuned class thresholds | 0.670 | 0.599 |

The per-class report uses the validation-tuned thresholds.

| Target | Precision | Recall | F1 | Positive images |
| --- | ---: | ---: | ---: | ---: |
| Maize | 1.000 | 1.000 | 1.000 | 13 |
| Meadow-grass | 0.341 | 0.938 | 0.500 | 16 |
| Creeping thistle | 0.342 | 1.000 | 0.510 | 13 |

Images can contain multiple targets. The tuned configuration achieved high recall for both weed classes, with precision near 0.34 for each. False-positive class predictions can generate prompts for plants that are absent.

## Segmentation validation ablation

The four variants share the classifier, class thresholds, CAM threshold, and 44 validation images. For each foreground class, IoU is computed by pooling pixel intersections and unions across the split. Foreground mIoU is the unweighted mean of the three class IoUs. Background is excluded from the class average, and non-target or unidentified pixels marked with the ignore label are excluded from all pixel counts.

| Method | Maize IoU | Meadow-grass IoU | Creeping thistle IoU | Foreground mIoU |
| --- | ---: | ---: | ---: | ---: |
| Grad-CAM | 1.76% | 0.34% | 17.43% | 6.51% |
| Grad-CAM + SAM 2 | 0.63% | 0.17% | 18.73% | 6.51% |
| Grad-CAM + ExG | **11.77%** | **0.93%** | 26.44% | **13.05%** |
| Grad-CAM + SAM 2 + ExG | 8.18% | 0.43% | **29.85%** | 12.82% |

Adding ExG to Grad-CAM increased validation mIoU by 6.54 percentage points. SAM 2 increased Creeping thistle IoU in both comparisons. Its mIoU tied the Grad-CAM baseline at the reported precision and was 0.23 percentage points lower when both variants used ExG.

## Limitations

- **Prompt quality.** Grad-CAM localizes features that drive the classifier and can miss plant boundaries. The validation notebook documents cases where imprecise prompts led SAM 2 to segment background. Errors in class prediction can also suppress or mislabel masks.
- **Color sensitivity.** A fixed ExG threshold can discard shaded or less-green plant pixels and retain other green regions.
- **Scope of the evidence.** The experiment uses 300 images and three targets. Meadow-grass IoU stayed below 1% across all validation variants. Performance on unseen sessions remains an open question for segmentation.
- **Evaluation granularity.** SAM 2 proposes object masks, which are fused into semantic maps. The reported pixel metrics measure class overlap. Individual plant separation has yet to be evaluated.

## Reproduction

The workflow starts from the saved ResNet18 checkpoint and original splits. It requires access to the WeedSAM Data bundle and the `02_gradcam_sam2` notebook.

1. **Prepare the inputs.** Attach the WeedSAM Data bundle to a Kaggle GPU notebook. Set `PROJECT_DIR` to the mounted `WeedSAM` folder. Use the following directory layout.

   ```text
   WeedSAM/
   ├── data/
   │   ├── images/
   │   └── labelIds/
   └── results/
       ├── config.json
       ├── train.csv
       ├── val.csv
       ├── test.csv
       └── weedsam_resnet18.pth
   ```

2. **Restore the models.** Run the setup cells in `02_gradcam_sam2` with PyTorch, torchvision, NumPy, pandas, Pillow, OpenCV, Matplotlib, and tqdm available. The notebook installs `grad-cam` and SAM 2 and downloads `sam2.1_hiera_tiny.pt` on first use. The saved ResNet18 checkpoint is from epoch 8. [SAM 2 setup and checkpoint instructions](https://github.com/facebookresearch/sam2)
3. **Review the classification run.** The saved `results/test_comparison.csv` and `results/test_classification_report.csv` contain the aggregate and per-class test metrics. To reproduce predictions with the tuned thresholds, score the images in `results/test.csv` with the restored classifier, apply sigmoid and the thresholds from `config.json`, and compare the resulting binary predictions with the three image-level target columns.
4. **Reproduce the validation ablation.** Run the Grad-CAM threshold sweep, SAM 2 mask generation, and ExG sweep on `val.csv`. Write generated artifacts to `/kaggle/working/weedsam_results/`. If the CAM sweep cells point their CSV and JSON writes at the Kaggle input mount, redirect those writes to the working directory. The expected outputs include `validation_segmentation_results.csv` and `vegetation_filter_results.csv`.
5. **Complete the pending segmentation test.** A previous Kaggle run generated all 184 masks, and a later session could not locate them. Recover the masks from a saved notebook version if available, or generate the four variants for `test.csv` with the frozen parameters above. Verify one matching PNG per test image in each of `cam`, `sam2`, `cam_exg`, and `sam2_exg` under `test_masks/`. Remap the ground truth to background (0), the three foreground IDs (1–3, in the saved class order), and ignore (255). Pool intersections, unions, and predicted and ground-truth pixel counts per class to compute IoU and Dice. Report the per-class scores and their foreground means in `final_test_results.csv`.
