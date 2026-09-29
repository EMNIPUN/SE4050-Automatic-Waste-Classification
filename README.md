# ♻️ Automatic Waste Classification Using Deep Learning

A deep learning-based image classification project developed for the **SE4050 – Deep Learning** module at the **Sri Lanka Institute of Information Technology (SLIIT)**.

The project investigates and compares four deep learning architectures for automatically classifying waste images into six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The four evaluated architectures are:

1. Custom CNN
2. ResNet50
3. MobileNetV2
4. EfficientNet-B0

---

## 📌 Project Overview

Efficient waste classification is an important component of automated waste management and recycling systems. Manual waste sorting can be time-consuming, labour-intensive, and prone to human error.

This project investigates the use of deep learning for automatic image-based waste classification.

The study compares a **Custom CNN trained from scratch** with three established deep learning architectures:

- **ResNet50**
- **MobileNetV2**
- **EfficientNet-B0**

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrices
- Model complexity
- Training behaviour
- Computational requirements

---

## 🎯 Project Objectives

The main objectives of this project are to:

- Analyze and preprocess a public waste image dataset.
- Identify and address duplicate and conflicting images.
- Prevent duplicate-related data leakage between dataset partitions.
- Develop a Custom CNN as a baseline model.
- Implement ResNet50, MobileNetV2, and EfficientNet-B0 models.
- Evaluate all models using multiple classification metrics.
- Compare model performance, complexity, and computational requirements.
- Analyze classification errors, overfitting, generalization, and limitations.

---

## 👥 Team Members

| Member | Student ID | Model / Contribution |
|---|---|---|
| Member 1 | IT23283930 | Custom CNN / EDA & Preprocessing |
| Member 2 | [Student ID] | ResNet50 |
| Member 3 | [Student ID] | MobileNetV2 |
| Member 4 | [Student ID] | EfficientNet-B0 |

> Replace the placeholders with the actual names and student IDs before submission.

---

# 📊 Dataset

## Dataset Source

The project uses the **Garbage Dataset Classification** dataset available on Kaggle.

Dataset:

**Garbage Dataset Classification – Kaggle**

The dataset can be accessed from:

`https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification`

The dataset is not stored directly in this repository because of its size.

---

## Dataset Classes

The original dataset contains **13,901 RGB images** belonging to six classes.

| Class | Original Images |
|---|---:|
| Cardboard | 2,214 |
| Glass | 2,500 |
| Metal | 2,084 |
| Paper | 2,315 |
| Plastic | 2,288 |
| Trash | 2,500 |
| **Total** | **13,901** |

The original images have a resolution of **256 × 256 pixels**.

---

# 🔍 Exploratory Data Analysis

Before model development, exploratory data analysis and data-quality checks were performed.

The analysis included:

- Class distribution analysis
- Image inspection
- Image dimension verification
- Corrupted image detection
- Exact duplicate detection
- Cross-class duplicate detection
- Data leakage analysis

### Duplicate Analysis

The original dataset contained:

| Measurement | Value |
|---|---:|
| Total images | 13,901 |
| Unique image contents | 11,657 |
| Duplicate files beyond first occurrence | 2,244 |
| Within-class duplicate groups | 2,242 |
| Cross-class conflicting duplicate groups | 2 |

Four files were associated with cross-class conflicting duplicates.

These conflicting images were removed.

### Clean Dataset

After cleaning:

**Total images: 13,897**

| Class | Clean Images |
|---|---:|
| Cardboard | 2,214 |
| Glass | 2,498 |
| Metal | 2,083 |
| Paper | 2,315 |
| Plastic | 2,287 |
| Trash | 2,500 |
| **Total** | **13,897** |

---

# 🔀 Dataset Splitting

A **group-aware stratified splitting strategy** was used.

Exact duplicate images were grouped using image-content hashes so that identical image content could not appear in multiple dataset partitions.

This reduces the risk of duplicate-related data leakage.

### Final Dataset Split

| Split | Groups | Images |
|---|---:|---:|
| Training | 8,158 | 9,692 |
| Validation | 1,748 | 2,114 |
| Test | 1,749 | 2,091 |
| **Total** | **11,655** | **13,897** |

Verification confirmed:

```text
Duplicate groups across splits: 0
```

The generated split information is stored in:

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

---

# 🧹 Data Preprocessing

The common preprocessing workflow includes:

1. Loading RGB images.
2. Resizing images to the required model input size.
3. Converting images to floating-point representation.
4. Applying the required input normalization/preprocessing.
5. Applying data augmentation to the training set.
6. Creating batched TensorFlow datasets.

For the models using the 224 × 224 pipeline, the primary input configuration was:

```text
Image Size: 224 × 224
Channels: RGB
Batch Size: 32
```

ResNet50 used a separate configuration:

```text
Image Size: 256 × 256
Channels: RGB
Batch Size: 16
```

> Architecture-specific preprocessing is maintained in the corresponding model notebook.

---

## Data Augmentation

Training-only augmentation includes:

```python
tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])
```

Augmentation is applied only to training data.

Validation and test images are not randomly augmented.

---

# 🧠 Deep Learning Models

## 1. Custom CNN

The Custom CNN was developed from scratch and used as the baseline model.

### Architecture

```text
Input: 224 × 224 × 3

Conv2D (32, 3×3)
Batch Normalization
ReLU
MaxPooling

Conv2D (64, 3×3)
Batch Normalization
ReLU
MaxPooling

Conv2D (128, 3×3)
Batch Normalization
ReLU
MaxPooling

Conv2D (256, 3×3)
Batch Normalization
ReLU

Global Average Pooling

Dense (128)
ReLU
Dropout (0.5)

Dense (6)
Softmax
```

### Training Configuration

| Parameter | Value |
|---|---|
| Input size | 224 × 224 × 3 |
| Batch size | 32 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Loss | Sparse Categorical Cross-Entropy |
| Maximum epochs | 30 |
| Early stopping patience | 5 |
| LR reduction factor | 0.5 |
| LR reduction patience | 2 |
| Minimum learning rate | 1e-6 |

### Parameters

| Type | Count |
|---|---:|
| Total | 424,006 |
| Trainable | 423,046 |
| Non-trainable | 970 |

---

## 2. ResNet50

ResNet50 uses residual connections to support training of deeper convolutional networks.

The implementation was adapted for six-class waste classification.

### Configuration

| Parameter | Value |
|---|---|
| Input size | 256 × 256 × 3 |
| Batch size | 16 |
| Number of classes | 6 |
| Head-training epochs | 15 |
| Fine-tuning epochs | 7 |
| Random seed | 42 |

### Parameters

| Type | Count |
|---|---:|
| Total | 23,850,758 |
| Trainable | 15,216,518 |
| Non-trainable | 8,634,240 |

See the ResNet50 notebook for the complete backbone, classification head, and fine-tuning implementation.

---

## 3. MobileNetV2

MobileNetV2 is a lightweight CNN architecture designed to provide effective image classification with relatively low computational complexity.

It uses mechanisms including:

- Depthwise separable convolutions
- Inverted residual blocks
- Linear bottlenecks

### Configuration

| Parameter | Value |
|---|---|
| Input size | 224 × 224 × 3 |
| Batch size | 32 |
| Number of classes | 6 |
| Epochs trained | 18 |

### Parameters

| Type | Count |
|---|---:|
| Total | 2,270,790 |
| Trainable | 1,536,646 |
| Non-trainable | 734,144 |

---

## 4. EfficientNet-B0

EfficientNet-B0 was implemented using a two-stage training strategy consisting of initial head training followed by fine-tuning.

### Configuration

| Parameter | Value |
|---|---|
| Input size | 224 × 224 × 3 |
| Batch size | 32 |
| Number of classes | 6 |
| Head-training epochs | Up to 20 |
| Head learning rate | 0.001 |
| Fine-tuning epochs | Up to 10 |
| Fine-tuning learning rate | 1e-5 |
| Fine-tuning point | Last 30 layers |
| Random seed | 42 |

### Parameters

| Type | Count |
|---|---:|
| Total | 4,057,257 |
| Trainable | 1,503,846 |
| Non-trainable | 2,553,411 |

---

# 📈 Experimental Results

The models were evaluated on the unseen test set.

## Overall Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Custom CNN | 67.24% | 67.98% | 67.24% | 66.94% | 92.70% |
| MobileNetV2 | 88.09% | 88.29% | 88.09% | 88.12% | 98.73% |
| EfficientNet-B0 | 90.15% | 90.30% | 90.15% | 90.16% | 99.08% |
| ResNet50 | 92.59% | 92.79% | 92.59% | 92.60% | 99.48% |

---

## Validation Performance

| Model | Best Validation Accuracy |
|---|---:|
| Custom CNN | 71.19% |
| MobileNetV2 | 87.84% |
| ResNet50 | 94.32% |
| EfficientNet-B0 | See model notebook/results |

---

## Model Complexity

| Model | Total Parameters | Trainable Parameters |
|---|---:|---:|
| Custom CNN | 424,006 | 423,046 |
| MobileNetV2 | 2,270,790 | 1,536,646 |
| EfficientNet-B0 | 4,057,257 | 1,503,846 |
| ResNet50 | 23,850,758 | 15,216,518 |

---

## Training Time

| Model | Training Time |
|---|---:|
| Custom CNN | 118.33 minutes |
| MobileNetV2 | 78.31 minutes |
| EfficientNet-B0 | 336.13 minutes |
| ResNet50 | Not recorded |

> Training-time comparisons should be interpreted carefully because model configurations, input sizes, batch sizes, and training strategies differ.

---

# 🔬 Key Observations

The experiments produced several important observations:

- The Custom CNN provided the baseline performance.
- MobileNetV2 substantially improved classification performance while maintaining a relatively small model size.
- EfficientNet-B0 provided further improvement with a moderate parameter count.
- ResNet50 produced the highest test metrics in these experiments but also had the largest parameter count.
- Paper and cardboard were recurring sources of confusion.
- Some models also confused visually related glass, plastic, and metal samples.
- Test-set performance does not necessarily represent performance on arbitrary real-world images with different backgrounds, lighting, orientations, and object conditions.

---

# 📁 Project Structure

```text
SE4050-Automatic-Waste-Classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_CNN.ipynb
│   ├── 03_ResNet50.ipynb
│   ├── 04_MobileNetV2.ipynb
│   └── 05_EfficientNetB0.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── results/
│   ├── figures/
│   ├── tables/
│   ├── metrics/
│   └── models/
│
└── report/
```

> Some generated directories may not be included in Git if they contain large datasets or model files.

---

# ⚙️ Environment Setup

## Prerequisites

Recommended environment:

```text
Python 3.11
TensorFlow 2.18
Conda / Miniconda
Jupyter Notebook or JupyterLab
```

---

## 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>

cd SE4050-Automatic-Waste-Classification
```

Replace `<YOUR-GITHUB-REPOSITORY-URL>` with the final GitHub repository URL.

---

## 2. Create a Conda Environment

```bash
conda create -n waste-dl python=3.11
```

Activate it:

```bash
conda activate waste-dl
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Register the Jupyter Kernel

```bash
python -m ipykernel install --user --name waste-dl --display-name "Python (waste-dl)"
```

Start Jupyter:

```bash
jupyter notebook
```

Select:

```text
Python (waste-dl)
```

as the notebook kernel.

---

# 📦 Dependencies

The project uses the following major dependencies:

```text
numpy==2.0.2
pandas==3.0.6
matplotlib==3.11.2
seaborn==0.13.2
scikit-learn==1.9.1
tensorflow==2.18.0
Pillow==12.3.0
jupyter==1.1.1
ipykernel==6.29.5
kaggle==2.2.4
```

For the complete dependency list, see:

```text
requirements.txt
```

---

# 📥 Dataset Setup

The dataset is not committed directly to GitHub.

Download the dataset from Kaggle:

`https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification`

After downloading, extract the dataset according to the structure described in:

```text
data/README.md
```

The raw dataset should remain excluded from Git using `.gitignore`.

---

## Optional: Kaggle CLI

If the Kaggle CLI is configured on your machine, the dataset can also be downloaded using the Kaggle command-line interface.

First verify Kaggle:

```bash
kaggle --version
```

Follow the current Kaggle instructions for authentication and dataset downloading.

---

# ▶️ Running the Project

The recommended execution order is:

### Step 1 — EDA and Preprocessing

Open:

```text
notebooks/01_EDA_Preprocessing.ipynb
```

Run the notebook to perform:

- Dataset inspection
- Class analysis
- Duplicate analysis
- Data cleaning
- Group-aware splitting
- Preprocessing verification

This generates the processed split CSV files.

---

### Step 2 — Custom CNN

Open:

```text
notebooks/02_CNN.ipynb
```

This notebook contains:

- Custom CNN architecture
- Model training
- Validation analysis
- Test evaluation
- Classification report
- Confusion matrix
- Training curves
- Prediction demo

---

### Step 3 — ResNet50

Open:

```text
notebooks/03_ResNet50.ipynb
```

Run the notebook to train/evaluate the ResNet50 implementation.

---

### Step 4 — MobileNetV2

Open:

```text
notebooks/04_MobileNetV2.ipynb
```

Run the notebook to train/evaluate the MobileNetV2 implementation.

---

### Step 5 — EfficientNet-B0

Open:

```text
notebooks/05_EfficientNetB0.ipynb
```

Run the notebook to train/evaluate the EfficientNet-B0 implementation.

---

# 📊 Generated Results

Experimental outputs are stored under:

```text
results/
```

Examples include:

```text
results/
├── figures/
│   ├── custom_cnn_confusion_matrix.png
│   ├── custom_cnn_accuracy_curve.png
│   └── custom_cnn_loss_curve.png
│
├── metrics/
│   └── custom_cnn_results.csv
│
└── models/
    └── custom_cnn.keras
```

Additional model-specific results are stored in their corresponding results locations.

---

# 🎲 Reproducibility

A random seed of:

```python
RANDOM_SEED = 42
```

was used where applicable.

For example:

```python
import random
import numpy as np
import tensorflow as tf

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
```

Environment information:

```text
Python: 3.11.16
TensorFlow: 2.18.0
Random Seed: 42
Number of Classes: 6
```

Because deep learning operations can depend on hardware, GPU libraries, and nondeterministic operations, exact results may vary slightly between environments.

---

# 📏 Evaluation Metrics

The following metrics were used to evaluate classification performance:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Class-wise classification report

Weighted metrics were used for overall precision, recall, and F1-score comparisons.

---

# ⚠️ Experimental Limitations

Several limitations should be considered when interpreting the results:

1. The models do not use completely identical architecture-specific configurations.
2. ResNet50 uses 256 × 256 inputs and batch size 16, while the other models primarily use 224 × 224 inputs and batch size 32.
3. Training strategies differ between architectures.
4. The dataset contains repeated image content, although group-aware splitting was used to prevent duplicate leakage between partitions.
5. The six dataset classes do not represent every possible type of real-world waste.
6. Real-world backgrounds, lighting conditions, object orientations, and mixed-material objects may reduce classification performance.
7. ResNet50 training time was not recorded, preventing a complete training-time comparison.

---

# 🚀 Future Improvements

Potential future improvements include:

- Increasing the diversity of real-world training images.
- Testing on an independent external waste dataset.
- Additional hyperparameter optimization.
- Stronger and more diverse augmentation strategies.
- Repeated experiments using multiple random seeds.
- Model pruning.
- Model quantization.
- Model compression.
- Deployment on resource-constrained or edge devices.
- Integration with automated waste-sorting systems.

---

# 📚 Academic Context

This project was developed as part of:

**SE4050 – Deep Learning**

Sri Lanka Institute of Information Technology (SLIIT)

Academic Year: **2026**

---

# 📄 License / Dataset Attribution

The dataset used in this project was obtained from Kaggle and remains subject to the terms and license specified by the original dataset provider.

Dataset:

**Garbage Dataset Classification**

`https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification`

The dataset itself is not redistributed through this repository.

---

# 👨‍💻 Contributors

This project was completed as a four-member group assignment.

| Member | Responsibility |
|---|---|
| Member 1 | EDA, preprocessing, duplicate analysis, Custom CNN |
| Member 2 | ResNet50 |
| Member 3 | MobileNetV2 |
| Member 4 | EfficientNet-B0 |

All members contributed to model evaluation, comparison, documentation, and final project analysis.

---

## Acknowledgement

This repository was developed for academic purposes as part of the SE4050 Deep Learning coursework at SLIIT.