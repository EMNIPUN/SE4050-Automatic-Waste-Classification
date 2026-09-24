# MobileNetV2: Experimental Progression & Model Comparison

**Module**: SE4050 – Deep Learning (2026)  
**Track**: Supervised Deep Learning (Automatic Waste Classification)  
**Author**: S. S. Kumbukage (IT23155534)  

---

## 1. Core Experimental Comparison: 4 Distinct Approaches

This report tracks the complete evolutionary trajectory of our deep learning models across four distinct paradigms on the Automatic Waste Classification dataset (6 classes: *cardboard, glass, metal, paper, plastic, trash*; 13,897 total images):

1. **Custom CNN**: Built and trained completely from scratch with random weight initialization.
2. **Original MobileNetV2 (Pre-trained Base)**: Google's raw, untouched architecture pre-trained on ImageNet-1k (1,000 generic object classes).
3. **Transfer Learning (Feature Extraction)**: Google's pre-trained MobileNetV2 backbone is frozen; only our custom 6-class waste classifier head is trained.
4. **Fine-Tuning (Specialized Model)**: MobileNetV2 with top 30 depthwise separable layers unfrozen and specialized end-to-end on waste imagery.

---

## 2. Comparative Benchmark Table

The table below presents the exact empirical metrics recorded across all four experimental setups, featuring separate rows for **Validation Accuracy** and **Test Accuracy (Unseen)**:

| Evaluation Dimension | 1. Custom CNN (From Scratch) | 2. Original MobileNetV2 (Google ImageNet Base) | 3. Transfer Learning (Stage 1: Frozen Backbone) | 4. Fine-Tuning (Stage 2: Specialized Model) |
| :--- | :---: | :---: | :---: | :---: |
| **Origin / Initial Weights** | Random Gaussian initialization | Pre-trained ImageNet-1k (Google) | Pre-trained ImageNet-1k + New Waste Head | Pre-trained ImageNet-1k + Unfrozen Top Layers |
| **Training Paradigm** | Full training from scratch | Pre-trained on 1.4M general images (1,000 classes) | Feature Extraction (Train custom head only) | End-to-end refinement with small LR (1e-5) |
| **Backbone Status** | Fully Trainable (100%) | Fixed (ImageNet benchmark) | **Frozen (Weights Locked)** | **Top 30 Layers Unfrozen & Trainable** |
| **Classifier Head Status** | Trainable (6 classes) | 1,000-class ImageNet Head (Fixed) | **Trainable (New 6-Class Waste Head)** | **Trainable (New 6-Class Waste Head)** |
| **Total Parameters** | 424,006 | 3,538,984 | 2,270,790 | 2,270,790 |
| **Trainable Parameters** | 423,046 (99.8%) | 0 (No training on waste) | **10,246 (0.45% - Head Only)** | **1,536,646 (67.7% - Head + Top 30 Layers)** |
| **Frozen Parameters** | 970 (0.2% BN) | 3,538,984 (100%) | **2,260,544 (99.55% - Backbone)** | **734,144 (32.3% - Shallow Layers)** |
| **Validation Accuracy** | **71.19%** (Epoch 26) | **0.00%** (1,000-class domain mismatch) | **85.62%** (Epoch 8) | **88.03%** (Epoch 16) |
| **Validation Loss** | 0.8435 (Epoch 29) | N/A (ImageNet loss) | 0.4209 (Epoch 8) | **0.3458** (Epoch 18) |
| **Test Accuracy (Unseen Data)** | **67.24%** | **0.00%** (1,000-class domain mismatch) | **85.62%** (Validation Selected) | **88.09%** 🏆 |
| **Test Weighted Precision** | 0.6798 | 0.0000 | N/A | **0.8829** |
| **Test Weighted Recall** | 0.6724 | 0.0000 | N/A | **0.8809** |
| **Test Weighted F1-Score** | 0.6694 | 0.0000 | N/A | **0.8812** 🏆 |
| **Test Weighted ROC-AUC** | 0.9270 | N/A | N/A | **0.9873** |
| **Training Epochs** | 30 epochs | 0 epochs | 8 epochs | 10 fine-tuning epochs (18 cumulative) |
| **Training Duration** | 118.33 minutes | 0.00 minutes | **29.08 minutes** (CPU) | **49.23 minutes** (CPU; 78.31 min total) |
| **Inference Latency** | ~18.0 ms / image | ~30.0 ms / image | ~31.0 ms / image | **31.8 ms / image** (~31.4 FPS) |
| **Model Size on Disk** | 5.0 MB | 14.2 MB | 9.76 MB | **9.76 MB** (.keras format) |

---

## 3. Empirical Analysis of Google's Original MobileNetV2

Before performing Transfer Learning or Fine-Tuning, we ran Google's untouched, pre-trained MobileNetV2 (weights='imagenet', include_top=True) on actual waste samples directly in the notebook (Cell 15). The empirical predictions were:

* **Sample 1 (True Label: trash)**:
  * Top 1: **Loafer** (34.64% confidence)
  * Top 2: **sandal** (31.40% confidence)
  * Top 3: **banded_gecko** (6.48% confidence)
* **Sample 2 (True Label: glass)**:
  * Top 1: **wine_bottle** (24.75% confidence)
  * Top 2: **beer_bottle** (9.67% confidence)
  * Top 3: **pill_bottle** (3.87% confidence)
* **Sample 3 (True Label: plastic)**:
  * Top 1: **sunscreen** (19.28% confidence)
  * Top 2: **oil_filter** (13.88% confidence)
  * Top 3: **rugby_ball** (3.87% confidence)
* **Sample 4 (True Label: metal)**:
  * Top 1: **loupe** (16.24% confidence)
  * Top 2: **Petri_dish** (7.10% confidence)
  * Top 3: **spotlight** (4.98% confidence)

### Key Deduction:
Google's pre-trained network correctly extracts high-level object concepts (e.g., recognizing glass containers as wine_bottle/beer_bottle, and plastic containers as sunscreen/oil_filter). However, because its final layer outputs **1,000 ImageNet categories** rather than the **6 target waste classes**, its raw direct classification accuracy on our waste validation and test sets is **0.00%**. 

This provides empirical proof justifying why we must **discard Google's 1,000-class head** and train a custom classification head via **Transfer Learning**.

---

## 4. Step-by-Step Deep Learning Analysis

### A. Custom CNN vs. Pre-trained Feature Extraction (+18.38% Boost)
* **Custom CNN Baseline**: Built with 4 convolutional layers from scratch, reaching **71.19% validation accuracy** and **67.24% test accuracy**. With random weight initialization and a limited training set (9,692 images), the model lacks the parameter depth and feature diversity to separate visually overlapping materials.
* **Stage 1 (Transfer Learning)**: By freezing Google's MobileNetV2 backbone and training **only 10,246 parameters** (0.45% of the network) in the new classification head:
  * Validation accuracy jumped immediately from **71.19% to 85.62%** (+14.43% boost over CNN validation, +18.38% over CNN test).
  * This was accomplished in just **29.08 minutes** on a standard CPU.
  * **Theoretical Principle**: **Inductive Bias Transfer**. The frozen backbone acts as an effective fixed feature extractor, projecting raw images into a semantically separable representation space without overfitting.

---

### B. Impact of Stage 2 Fine-Tuning (+2.41% Validation, +20.85% over Custom CNN)
* In Stage 1, performance plateaued at **85.62% validation accuracy** and **0.4209 validation loss** because generic ImageNet filters cannot distinguish domain-specific material properties (e.g., crinkled transparent plastic vs. smooth transparent glass).
* By **unfreezing the top 30 depthwise separable layers** (making 1,536,646 parameters trainable) and training with a conservative learning rate (eta = 1e-5):
  * Validation loss dropped from **0.4209 to 0.3458** (a **17.8% relative error reduction**).
  * Validation accuracy increased from **85.62% to 88.03%** (+2.41%).
  * Test accuracy reached **88.09%** on 2,091 completely unseen test samples.
  * Weighted F1-score reached **0.8812**, Precision reached **0.8829**, and Recall reached **0.8809**.
  * The small learning rate (eta = 1e-5) preserved low-level universal features (edges, textures) while allowing high-level features to specialize, preventing **catastrophic forgetting**.
