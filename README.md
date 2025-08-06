# MediCLIP - Medical Image Classification using CLIP

**MediCLIP** is an advanced medical image classification system that leverages CLIP (Contrastive Language–Image Pretraining) to differentiate between medical and non-medical images. It achieves high performance through zero-shot learning, removing the dependency on large annotated datasets while maintaining robust accuracy across diverse medical imaging types.

---

## Project Resources

> Note: The dataset and trained model are large in size and cannot be hosted directly on GitHub. Please use the following external links to access them.

* **Dataset (Custom-Created and Curated):**
  [MediCLIP - Medical Image Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/tiwariar273/mediclip-medical-image-classification-dataset)

* **Trained Model (CLIP ViT-B/32 - Saved Weights and Configuration):**
  [Google Drive - MediCLIP Saved Model](https://www.kaggle.com/models/tiwariar273/mediclip-medical-image-classification-using-clip/)

---

## Overview

MediCLIP demonstrates 96.16% test accuracy using a zero-shot classification pipeline based on OpenAI's CLIP ViT-B/32 architecture. By leveraging natural language supervision, MediCLIP avoids the traditional cost of dataset annotation, while achieving robust generalization across varied image modalities such as X-rays, MRIs, CT scans, and more.

---

## Methodology and Model Selection

### 1. Evaluation of Alternative Approaches

Several techniques were considered and empirically evaluated before finalizing the use of CLIP.

#### 1.1 One-Class Classification Models

Medical images represent a unique semantic category, leading to an initial exploration of anomaly detection and one-class learning approaches:

* **Deep SVDD (Deep Support Vector Data Description):**
  Aimed to model the compact representation of medical images. However, it failed to generalize due to the high intra-class variability (e.g., differences between X-rays, MRIs, and ultrasounds).

* **One-Class SVM:**
  Performed poorly due to the high dimensionality of deep image embeddings, which led to weak class boundaries.

* **OC-NN (One-Class Neural Network):**
  Although it leveraged deep features, it required heavy hyperparameter tuning and still underperformed.

#### 1.2 Binary Classification Models

Supervised binary classifiers were also tested with labeled data:

* **ResNet-18 (Fine-Tuned):**
  Delivered reasonable performance but required large amounts of labeled data and significant training time.

* **EfficientNet-B0:**
  Despite being optimized for efficiency, it suffered from overfitting when trained on a limited dataset and also required extensive labeling.

---

### 2. Why CLIP Was Selected

The CLIP model (ViT-B/32) was chosen for the following advantages:

* **Zero-Shot Learning:**
  Allows classification based on text prompts without requiring labeled training data.

* **Generalization Across Modalities:**
  Effectively handles diverse medical imaging formats without the need for retraining or fine-tuning.

* **Multimodal Understanding:**
  Utilizes both visual and textual semantics, improving class separability and reducing false positives.

* **Scalability and Extensibility:**
  Supports rapid adaptation to new categories simply by introducing new descriptive text prompts.

---

## Dataset Summary

* **Total Images:** 14,668
* **Training Set:** 14,078 images (Medical: 6,827 | Non-Medical: 7,251)
* **Validation Set:** 2,769 images (Medical: 1,342 | Non-Medical: 1,427)
* **Test Set:** 2,789 images (Medical: 1,363 | Non-Medical: 1,426)

---

## Performance Metrics

### Training Set Evaluation

```
Overall Accuracy: 86.01%

Medical Class:
- Precision: 80%
- Recall: 95%
- F1-Score: 87%

Non-Medical Class:
- Precision: 94%
- Recall: 77%
- F1-Score: 85%
```

### Validation Set Evaluation

```
Overall Accuracy: 95.77%

Medical Class:
- Precision: 92%
- Recall: 100%
- F1-Score: 96%

Non-Medical Class:
- Precision: 100%
- Recall: 92%
- F1-Score: 96%
```

### Test Set Evaluation

```
Overall Accuracy: 96.16%

Medical Class:
- Precision: 93%
- Recall: 100%
- F1-Score: 96%
- Support: 1,363

Non-Medical Class:
- Precision: 100%
- Recall: 93%
- F1-Score: 96%
- Support: 1,426

Macro Average:
- Precision: 96%
- Recall: 96%
- F1-Score: 96%
```
---

## Observations and Key Insights

1. **High Precision for Non-Medical Class (100%)**: Minimizes the risk of incorrectly classifying non-medical content as medical, which is critical in clinical use cases.

2. **Perfect Recall for Medical Class (100%)**: Ensures all relevant medical images are correctly identified.

3. **Balanced Classification (F1-Score: 96% for both classes)**: Indicates excellent trade-off between precision and recall across both categories.

4. **Generalization Capabilities**: Despite a lower training accuracy (86.01%), the test accuracy (96.16%) confirms the model's strong generalization to unseen data.

---



---


