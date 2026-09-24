# MobileNetV2 Stage 2: Fine-Tuning & Model Selection Analysis

**Module**: SE4050 – Deep Learning (2026)  
**Track**: Supervised Deep Learning (Automatic Waste Classification)  
**Author**: S. S. Kumbukage (IT23155534)  
**Model**: MobileNetV2 (Top 30 Layers Unfrozen)  
**Stage**: Phase 2 — Fine-Tuning & Validation-Driven Selection  

---

## 1. Experimental Configuration & Hyperparameters

* **Backbone Status**: Top 30 layers **unfrozen** (shallow layers remain frozen)
* **Trainable Parameters**: 1,536,646 (increased from 10,246 in Stage 1)
* **Optimizer**: Adam (learning_rate = 1e-5, reduced 100x from Stage 1)
* **Loss Function**: Sparse Categorical Crossentropy
* **Batch Size**: 32
* **Epochs Trained**: Epochs 9 to 18 (10 fine-tuning epochs)
* **Hardware Mode**: CPU Execution (Elapsed Time: 2,953.53 seconds / ~49.23 minutes)
* **Saved Checkpoint**: src/models/mobilenetv2_finetuned_best.keras

---

## 2. Frozen vs. Fine-Tuned Performance Comparison

| Metric | Stage 1 (Frozen Backbone) | Stage 2 (Fine-Tuned) | Delta / Improvement |
| :--- | :---: | :---: | :--- |
| **Best Validation Loss** | 0.4209 | **0.3458** | **-0.0751** (17.8% relative error reduction) |
| **Best Validation Accuracy** | 85.62% | **88.03%** | **+2.41% absolute improvement** |
| **Final Training Loss** | 0.5208 | **0.3147** | -0.2061 (smooth optimization) |
| **Final Training Accuracy** | 81.94% | **88.91%** | +6.97% |
| **Trainable Parameters** | 10,246 | 1,536,646 | +1,526,400 parameters specialized |
| **Learning Rate** | 1e-3 (0.001) | 1e-5 (0.00001) | 100x smaller to prevent catastrophic forgetting |

---

## 3. Epoch-by-Epoch Fine-Tuning Trajectory

| Epoch | Train Loss | Val Loss | Train Accuracy | Val Accuracy | Learning Rate | Checkpoint Event |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **9** | 0.5185 | 0.4357 | 82.20% | 85.67% | 1e-5 | Checkpoint saved |
| **10** | 0.5056 | 0.4312 | 82.75% | 85.95% | 1e-5 | Checkpoint saved |
| **11** | 0.5009 | 0.4107 | 82.80% | 86.80% | 1e-5 | Checkpoint saved |
| **12** | 0.4659 | 0.3950 | 83.62% | 86.85% | 1e-5 | Checkpoint saved |
| **13** | 0.4248 | 0.3823 | 84.96% | 86.99% | 1e-5 | Checkpoint saved |
| **14** | 0.3925 | 0.3782 | 86.07% | 87.51% | 1e-5 | Checkpoint saved |
| **15** | 0.3685 | 0.3716 | 87.20% | 87.61% | 1e-5 | Checkpoint saved |
| **16** | 0.3668 | 0.3567 | 87.16% | 88.03% | 1e-5 | Checkpoint saved |
| **17** | 0.3499 | 0.3536 | 87.74% | 87.98% | 1e-5 | Checkpoint saved |
| **18** | **0.3147** | **0.3458** | **88.91%** | **87.84%** | **1e-5** | Best Val Loss Checkpoint |

---

## 4. Deep Learning Analysis & Discussion

### Observation A: Continuous Monotonic Loss Reduction
* **Behavior**: Across all 10 fine-tuning epochs, validation loss dropped on almost every single epoch (from 0.4357 down to 0.3458).
* **Theoretical Explanation**: This demonstrates that the learning rate of ^{-5}$ was mathematically well-conditioned. If the learning rate had been too high (e.g., ^{-3}$), gradient updates would have destabilized the pre-trained weights, causing loss divergence or erratic oscillations. The small learning rate allowed the convolutional filters to gently adapt their feature detectors to waste surfaces without losing general visual knowledge.

### Observation B: Breaking the 85.6% Ceiling
* **Behavior**: Stage 1 hit a plateau at 85.62%. Stage 2 pushed validation accuracy to **88.03%**.
* **Theoretical Explanation**: This proves that generic ImageNet representations alone are insufficient for fine-grained waste classification. Unfreezing the top 30 depthwise separable layers allowed the network to learn subtle textures specific to waste materials (e.g., specular reflection patterns on clear glass vs. polyethylene plastic, surface ridges on corrugated cardboard, and crinkled deformities in aluminum cans).

---

## 5. Model Selection Decision
Following our predefined experimental protocol:
* **Selected Winning Model**: **Stage 2 (Fine-Tuned)**
* **Selection Criterion**: Lower validation loss (.3458$ vs. .4209$) and superior validation accuracy (.03\%$ vs. .62\%$).
* **Checkpoint Restored**: src/models/mobilenetv2_finetuned_best.keras
