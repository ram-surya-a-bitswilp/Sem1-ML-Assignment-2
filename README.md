# Breast Cancer Wisconsin Diagnostic Classification

## a. Problem Statement
Breast cancer is one of the most common cancers diagnosed among women globally. Early and accurate diagnosis of breast mass tissue—classifying it as **Malignant** ($M$) or **Benign** ($B$)—is critical for effective clinical treatment and improving patient survival rates. 

The objective of this project is to build a modular, end-to-end Machine Learning pipeline and interactive Streamlit web application to evaluate multiple classification models on digitised fine needle aspirate (FNA) test samples, identify the optimal predictive model based on comprehensive evaluation metrics, and provide seamless automated diagnostic inference.

---

## b. Dataset Description
The model pipeline is trained and evaluated using the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

* **Total Samples:** 569 instances
* **Feature Count:** 30 continuous numerical features computed from digitized images of fine needle aspirates (FNA) of breast masses.
* **Feature Categories:** Features capture attributes of cell nuclei including `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave points`, `symmetry`, and `fractal dimension` across three statistical aggregates: **Mean**, **Standard Error (SE)**, and **Worst** (mean of the three largest values).
* **Target Variable:** `diagnosis`
  * `M`: Malignant (Binary class 1)
  * `B`: Benign (Binary class 0)

---

## c. Github Repository Link
**Repository Link:** [https://github.com/your-username/breast-cancer-diagnostic-ml](https://github.com/ram-surya-a-bitswilp/Sem1-ML-Assignment-2)

---

## d. Models Used

### ML Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9649 | 0.9960 | 0.9652 | 0.9649 | 0.9647 | 0.9245 |
| **Decision Tree** | 0.9298 | 0.9246 | 0.9298 | 0.9298 | 0.9298 | 0.8492 |
| **kNN** | 0.9561 | 0.9823 | 0.9569 | 0.9561 | 0.9558 | 0.9058 |
| **Naive Bayes** | 0.9211 | 0.9891 | 0.9211 | 0.9211 | 0.9204 | 0.8292 |
| **Random Forest (Ensemble)** | 0.9737 | 0.9929 | 0.9747 | 0.9737 | 0.9735 | 0.9442 |

---

### Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well as a linear baseline, achieving **96.49% accuracy** and the highest overall **AUC score (0.9960)**. This demonstrates that the scaled linear feature boundaries in the dataset are strong and linearly separable. |
| **Decision Tree** | Showed the lowest non-probabilistic metrics among non-probabilistic models with an **accuracy of 92.98%** and **AUC of 0.9246**. A single tree is prone to variance and slight overfitting compared to ensemble methods. |
| **kNN** | Delivered strong distance-based classification performance with **95.61% accuracy** and an **AUC of 0.9823**, proving effective after dynamic standard scaling of continuous features. |
| **Naive Bayes** | Recorded an **accuracy of 92.11%** and **MCC of 0.8292**, but retained a very high **AUC score of 0.9891**. The independence assumption slightly degrades point accuracy due to correlated cell features (e.g., radius, perimeter, and area), but rank probability separation remains strong. |
| **Random Forest (Ensemble)** | **Top Performing Model overall.** The ensemble of decision trees effectively reduced variance and captured non-linear interactions across nucleus properties, achieving the highest scores across **Accuracy (97.37%)**, **F1 (97.35%)**, and **MCC (0.9442)**. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** is the overall winner. It achieved the highest overall Accuracy (0.9737), Precision (0.9747), Recall (0.9737), F1-Score (0.9735), and Matthews Correlation Coefficient (0.9442), making it the most reliable model for minimizing critical diagnostic misclassifications. |
