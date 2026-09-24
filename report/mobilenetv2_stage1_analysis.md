# MobileNetV2 Stage 1: Feature Extraction Analysis

**Module**: SE4050 – Deep Learning (2026)  
**Track**: Supervised Deep Learning (Automatic Waste Classification)  
**Author**: S. S. Kumbukage (IT23155534)  
**Model**: MobileNetV2 (Pre-trained on ImageNet-1k)  
**Stage**: Phase 1 — Feature Extraction (Frozen Backbone)  

---

## 1. Experimental Configuration & Hyperparameters

* **Backbone**: MobileNetV2 (include_top=False, weights=imagenet)
* **Backbone Status**: **Frozen** (ase_model.trainable = False)
* **Classification Head**:
  Input (224x224x3) -> MobileNetV2 -> GlobalAvgPool2D -> BatchNorm -> Dropout(0.3) -> Dense(6, softmax)
* **Optimizer**: Adam (learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999)
* **Loss Function**: Sparse Categorical Crossentropy
* **Batch Size**: 32
* **Target Epochs**: 8
* **Dataset Splits**:
  * **Train**: 9,692 images (303 batches)
  * **Validation**: 2,114 images (67 batches)
  * **Test**: 2,091 images (66 batches, strictly unseen)
* **Random Seed**: 42
* **Hardware Mode**: CPU Execution (Elapsed Time: 1,744.93 seconds / ~29.08 minutes)
* **Saved Checkpoint**: src/models/mobilenetv2_stage1_best.keras

---

## 2. Parameter Accounting Summary

| Parameter Group | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Total Parameters** | **2,270,790** | 100.0% | Complete network size |
| **Trainable Parameters** | **10,246** | **0.45%** | Only the classification head (GlobalAvgPool + BN + Dense) |
| **Non-Trainable Parameters** | **2,260,544** | **99.55%** | Frozen pre-trained ImageNet convolutional layers |

> **Key Takeaway**: By updating only 10,246 parameters (less than 0.5% of the total network), the model reached an impressive **85.62% validation accuracy** within 8 epochs.

---

## 3. Epoch-by-Epoch Metric Log

| Epoch | Train Loss | Val Loss | Train Accuracy | Val Accuracy | Learning Rate | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.9516 | 0.4988 | 68.05% | 82.50% | 0.001 | Checkpoint saved |
| **2** | 0.6474 | 0.4765 | 77.55% | 83.87% | 0.001 | Checkpoint saved |
| **3** | 0.5709 | 0.4497 | 80.08% | 84.82% | 0.001 | Checkpoint saved |
| **4** | 0.5340 | 0.4463 | 81.31% | 84.91% | 0.001 | Checkpoint saved |
| **5** | 0.5204 | 0.4407 | 81.96% | 85.19% | 0.001 | Checkpoint saved |
| **6** | 0.5098 | 0.4652 | 81.91% | 84.82% | 0.001 | Val loss plateaued |
| **7** | 0.5180 | 0.4402 | 81.80% | 85.19% | 0.001 | Checkpoint saved |
| **8** | **0.5208** | **0.4209** | **81.94%** | **85.62%** | **0.001** | Best Model Checkpoint |

* **Best Validation Loss**: **0.42086** (Epoch 8)
* **Best Validation Accuracy**: **85.62%** (Epoch 8)

---

## 4. Deep Learning Analysis & Discussion

### Observation A: Validation Accuracy Exceeds Training Accuracy
In standard models trained from scratch, training accuracy is almost always higher than validation accuracy. However, in our Stage 1 results, validation accuracy (85.62%) was consistently 3.5% - 14% higher than training accuracy (81.94%). 

**Theoretical Explanation**:
1. **On-the-Fly Data Augmentation Penalty**: Training images undergo random flips, translations (+/- 10%), zooms (+/- 10%), and rotations (+/- 10%). These perturbations create artificially distorted, challenging samples during training. In contrast, the validation set consists of clean, unperturbed images, making evaluation comparatively easier.
2. **Dropout Regularization Dynamics**: The classification head incorporates Dropout(0.3). During training, 30% of activations are randomly dropped per forward pass, artificially reducing network capacity. During validation, dropout is deactivated (	raining=False), allowing all neurons to work cooperatively at full ensemble capacity.

---

### Observation B: Rapid Initial Convergence (Epochs 1 to 4)
* **Behavior**: Loss decreased steeply from 0.9516 to 0.5340 in just 4 epochs, and accuracy jumped from 68.05% to 81.31%.
* **Theoretical Explanation**: This illustrates the power of **inductive bias transfer**. The pre-trained MobileNetV2 backbone already retains general visual representations (edge filters, corner detectors, textures, and shape manifolds) learned from 1.4 million ImageNet images. The linear classifier head only had to discover the hyperplanes separating these existing representations into 6 waste classes.

---

### Observation C: Performance Plateau Around ~85% (Epochs 5 to 8)
* **Behavior**: From Epoch 5 to 8, training accuracy hovered around 81.9% and validation accuracy stabilized between 85.1% and 85.6%.
* **Theoretical Explanation**: This plateau marks the **representational ceiling** of feature extraction. Because the convolutional backbone remained frozen, the network could only rely on generic features learned from ImageNet. Generic features cannot fully capture subtle domain-specific waste characteristics (such as transparent plastic reflections vs. clear glass bottles, or crumpled aluminum foil vs. textured cardboard).

---

## 5. Technical Justification for Stage 2 (Fine-Tuning)

To break through the 85.6% performance ceiling and achieve >= 90% test accuracy:
1. **Unfreeze Top Backbone Layers**: Unfreeze the top 30 layers of the MobileNetV2 backbone (keeping shallow low-level feature extractors frozen).
2. **Reduced Learning Rate**: Lower the learning rate by two orders of magnitude (eta = 1e-5) to prevent large gradient updates from destroying pre-trained weights (**catastrophic forgetting**).
3. **Domain Specialization**: Allow the top depthwise separable convolutions to specialize their receptive fields specifically to waste textures and material surfaces.
