
# Aerial Landscape Images Classification via machine learning and deep learning

This repository contains a collection of notebooks and scripts for classifying Aerial Landscape Images using handcrafted feature extraction (SIFT, LBP), classical machine learning classifiers (SVM, Random Forest) and deep learning classifiers(EfficientNetB0, ResNet-18, VGG-16), with optional support for object detection and visualization. The dataset includes 15 categories of landscapes, each with 800 images at a resolution of 256×256 pixels, totaling 12,000 images.

Categories include:
Agriculture, Airport, Beach, City, Desert, Forest, Grassland, Highway, Lake, Mountain, Parking, Port, Railway, Residential, River.

---

## Notebooks Overview

| Notebook Name           | Feature Type       | Classifier(s)                  | Key Functions                                                                 |
|-------------------------|--------------------|--------------------------------|--------------------------------------------------------------------------------|
| `SIFT+SVM.ipynb`        | SIFT               | Support Vector Machine         | `extract_sift_descriptors`, `build_codebook`, `compute_bow_histogram`         |
| `SIFT+RF.ipynb`         | SIFT               | Random Forest                  | Same as above                                                                 |
| `LBP+RF.ipynb`          | LBP                | Random Forest                  | `compute_lbp_histogram`, `extract_lbp_features`, `augment_image`              |
| `LBP+SVM.ipynb`         | LBP                | SVM                            | Same as above + `create_imbalanced_dataset`                                   |
| `ML.ipynb`              | SIFT (BoW)         | KNN / Logistic Regression      | `initImg`, `get_descriptors`, `build_bow_histogram`, `SMOTE`                 |
| `efficientnetb0.ipynb`     | Image Processing | EfficientNetB0                 | `get_image_paths_and_labels`, `preprocess`, `add_noise`                       |
| `ResNet.ipynb`          | Data Resampling     | ResNet-18                      | `make_imbalanced_dataset`, `print_class_distribution`, `__getitem__`          |

Important: `ML.ipynb` should be run on colab while the rest can be run on kaggle

---

## Classical Machine Learning

### Classical Machine Learning Classifier Methodology

Each notebook typically follows this pipeline:

1. **Image Loading**  
   Images are read from directories organized by class labels.

2. **Feature Extraction**  
   - **SIFT (Scale-Invariant Feature Transform)** for keypoint-based descriptors  
   - **LBP (Local Binary Pattern)** for texture-based descriptors

3. **Vectorization**  
   - SIFT features are converted into fixed-size histograms using **Bag-of-Words (BoW)** with KMeans.  
   - LBP features use direct histogram aggregation.

4. **Classification**  
   - Trained using **SVM**, **Random Forest**, or **Logistic Regression**  
   - Optional **SMOTE** is applied to handle class imbalance.

5. **Evaluation**  
   Includes metrics such as Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

---

### Classical Machine Learning Classifier Results & Comparisons

Each notebook independently reports:
- Classification accuracy
- Confusion matrix
- Macro-averaged precision, recall, F1-score

These results can be compared to evaluate the impact of different feature-classifier combinations.

---

## Deep Learning Classifier

### Deep Learning Classifier Model Description

There are three Deep Learning models in the `models/` folder:

- **EfficientNetB0**: A lightweight pretrained architecture with compound scaling; achieves strong generalization without data augmentation.
- **ResNet-18**: A residual network with 18 layers; uses skip connections to ease gradient flow and boost performance.  

---

### Deep Learning Model Performance Summary

| Model         | Accuracy     | F1-Score     | Augmentation Used | Notes                                                                                     |
|---------------|--------------|--------------|-------------------|-------------------------------------------------------------------------------------------|
| EfficientNetB0| 94%          | > 90% (avg)  | ✅                | *river* recall = 0.87; strong generalization with pretrained backbone                     |
| ResNet-18     | 96.88%       | 0.9688       | ✅                | High overall accuracy and balanced class performance                                      |

Note the training and test time are benchmarked using a T4*2 GPU. 

---