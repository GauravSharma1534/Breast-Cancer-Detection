#  Breast Cancer Detection using Machine Learning

> **Gaurav Sharma | 23BDA70050 | Chandigarh University**  
> Python · Scikit-learn · Pandas · Matplotlib · 2025

## Project

Breast cancer is one of the most common cancers in women worldwide. Early detection can save lives. In this project, I built a machine learning pipeline that classifies breast tumours as **Benign** or **Malignant** using the Wisconsin Breast Cancer Dataset.

I compared three machine learning models and also applied ROC-curve threshold tuning to reduce the number of missed malignant cases (false negatives) by around **18%**.

This is my B.Tech Data Science project built as part of my coursework at Chandigarh University.

---

##  Dataset

| Property | Details |
|----------|---------|
| Name | Wisconsin Breast Cancer Dataset |
| Source | `sklearn.datasets.load_breast_cancer()` |
| Total Samples | 569 |
| Features | 30 (numerical) |
| Benign | 357 (62.7%) |
| Malignant | 212 (37.3%) |
| Missing Values | None |

The 30 features are computed from digitised images of fine needle aspirate (FNA) biopsies. They include measurements like radius, texture, perimeter, area, smoothness, compactness, concavity etc. — each given as mean, standard error, and worst value.

---

#  Project Objectives

The primary objectives of this project are:

* Detect breast cancer using Machine Learning techniques.
* Classify tumors as Benign or Malignant.
* Compare multiple classification algorithms.
* Analyze model performance using various evaluation metrics.
* Minimize False Negative predictions.
* Improve reliability through Cross Validation.
* Visualize important patterns and insights using charts.
* Demonstrate the practical application of AI in healthcare.

---

#  Problem Statement

Traditional cancer diagnosis relies heavily on expert medical interpretation and laboratory procedures. While highly effective, manual diagnosis can be time-consuming and may be influenced by human error.

Machine Learning offers an opportunity to assist healthcare professionals by:

* Automating preliminary screening.
* Providing fast predictions.
* Identifying important diagnostic features.
* Supporting decision-making processes.

The goal is to build a predictive system capable of accurately classifying breast tumors using patient biopsy measurements.

---

#  Dataset Information

## Wisconsin Breast Cancer Dataset

| Property       | Details                               |
| -------------- | ------------------------------------- |
| Dataset Name   | Wisconsin Breast Cancer Dataset       |
| Source         | sklearn.datasets.load_breast_cancer() |
| Total Samples  | 569                                   |
| Features       | 30                                    |
| Target Classes | Benign, Malignant                     |
| Missing Values | None                                  |

### Class Distribution

| Class     | Samples | Percentage |
| --------- | ------- | ---------- |
| Benign    | 357     | 62.7%      |
| Malignant | 212     | 37.3%      |

The dataset contains measurements computed from digitized images of Fine Needle Aspirate (FNA) biopsies.

Examples of features include:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Symmetry
* Fractal Dimension

Each characteristic is represented through:

* Mean Value
* Standard Error
* Worst Value

resulting in 30 numerical features.

---

##  Models Used

I trained and compared 3 models:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Neg. |
|-------|----------|-----------|--------|----------|---------|-----------|
| Logistic Regression | **98.25%** | 98.61% | 98.61% | 98.61% | 99.54% | 1 |
| SVM (RBF Kernel) | 97.37% | 98.59% | 97.22% | 97.90% | **99.57%** | 2 |
| Random Forest | 95.61% | 95.89% | 97.22% | 96.55% | 99.32% | 3 |
| RF (ROC-Optimised) | 96.49% | — | Higher | — | — | ↓ ~18% FN |

---



##  How to Run

**Step 1 — Clone the repo**
```bash
git clone https://github.com/GauravSharma1534/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
```

**Step 2 — Install required libraries**
```bash
pip install scikit-learn pandas matplotlib seaborn numpy
```

**Step 3 — Run the pipeline**
```bash
python breast_cancer_detection.py
```

Running this will:
- Load the dataset and print basic info
- Train all 3 models and print results
- Apply ROC-threshold tuning on Random Forest
- Save 5 charts to the `charts/` folder
- Run 5-Fold Cross-Validation and show score distribution

---



---

##  Key Findings

- **Logistic Regression** gave the best accuracy of **98.25%** — even though it's the simplest model. This shows the dataset is largely linearly separable after scaling.
- **SVM** achieved the highest ROC-AUC of **99.57%**, making it best for threshold-based clinical tuning.
- **Random Forest** provides feature importances — the most useful features are *worst concave points*, *worst area*, and *worst radius*.
- **ROC-threshold tuning** (Youden's J statistic) on Random Forest reduced false negatives by **~18%**, which is important in medical settings where missing a cancer case is dangerous.
- **5-Fold Cross-Validation** shows all models generalise well with standard deviation < 0.01.

---

##  Technologies 

- **Python 3.x**
- **scikit-learn** — models, preprocessing, metrics
- **pandas** — data loading, manipulation
- **matplotlib + seaborn** — all visualisations
- **numpy** — numerical operations

---



---



#  Machine Learning Workflow

The project follows the complete Machine Learning lifecycle:

### 1. Data Loading

Dataset imported directly from Scikit-Learn.

### 2. Exploratory Data Analysis (EDA)

Performed to understand:

* Class distribution
* Feature relationships
* Feature correlations
* Outlier patterns

### 3. Data Preprocessing

* Train-Test Split
* Feature Scaling using StandardScaler
* Data Validation

### 4. Model Training

Three classification algorithms are trained:

* Logistic Regression
* Support Vector Machine (RBF Kernel)
* Random Forest Classifier

### 5. Model Evaluation

Performance measured using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

### 6. ROC Threshold Optimization

Applied to Random Forest to reduce False Negatives.

### 7. Cross Validation

5-Fold Cross Validation used to evaluate generalization performance.

---

#  Models Used

## Logistic Regression

A linear classification algorithm widely used in healthcare applications.

Advantages:

* Fast training
* Highly interpretable
* Works well on linearly separable datasets

---

## Support Vector Machine (RBF)

Uses nonlinear decision boundaries to classify complex data patterns.

Advantages:

* High accuracy
* Strong generalization
* Excellent ROC-AUC performance

---

## Random Forest

An ensemble learning method based on multiple decision trees.

Advantages:

* Robust performance
* Handles nonlinear relationships
* Provides feature importance scores

---

#  Model Performance Comparison

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC  | False Negatives |
| ------------------- | -------- | --------- | ------ | -------- | -------- | --------------- |
| Logistic Regression | 98.25%   | 98.61%    | 98.61% | 98.61%   | 99.54%   | 1               |
| SVM (RBF)           | 97.37%   | 98.59%    | 97.22% | 97.90%   | 99.57%   | 2               |
| Random Forest       | 95.61%   | 95.89%    | 97.22% | 96.55%   | 99.32%   | 3               |
| ROC Optimized RF    | 96.49%   | Improved  | Higher | Improved | Improved | Reduced         |

---

#  Visualization Outputs

The project automatically generates multiple visualizations.

### 1. Exploratory Data Analysis

Includes:

* Class Distribution
* Correlation Analysis
* Boxplots

### 2. Model Comparison

Displays:

* Accuracy
* Precision
* Recall
* F1 Score

### 3. ROC Curves

Visual comparison of classifier performance.

### 4. Confusion Matrices

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### 5. Feature Importance

Displays Top 15 Important Features identified by Random Forest.

### 6. Cross Validation Results

Illustrates model stability using boxplots.

---

#  Project Structure

```text
Breast-Cancer-Detection/
│
├── breast_cancer_detection.py
│
├── charts/
│   ├── 01_eda_overview.png
│   ├── 02_model_comparison.png
│   ├── 03_confusion_matrices.png
│   ├── 04_feature_importance.png
│   └── 05_cv_scores.png
│
├── Project_Report_Gaurav.pdf
├── Breast_Cancer_Detection.pptx
├── Demo_Video.html
├── Output_Demo_Video.html
└── README.md
```

---

#  Installation

Clone Repository

```bash
git clone https://github.com/GauravSharma1534/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
```

Install Dependencies

```bash
pip install scikit-learn pandas matplotlib seaborn numpy
```

Run Project

```bash
python breast_cancer_detection.py
```

---

#  Expected Output

The program will:

* Load Dataset
* Perform Data Analysis
* Train Models
* Evaluate Performance
* Generate Charts
* Display Results
* Save Visualizations

---
##  Output Charts

The script automatically generates these 5 charts:

1. **EDA Overview** — class balance bar chart, feature correlation with target, mean radius boxplot by class
2. **Model Comparison** — grouped bar chart (Acc/Prec/Recall/F1) + ROC curves for all 3 models
3. **Confusion Matrices** — heatmaps showing TP, TN, FP, FN for each model
4. **Feature Importance** — top 15 most important features from Random Forest
5. **Cross-Validation** — boxplot of 5-fold accuracy scores per model


#  Key Findings

### Logistic Regression

Achieved the highest overall accuracy of 98.25%.

This indicates that the dataset becomes highly separable after feature scaling.

### Support Vector Machine

Achieved the highest ROC-AUC score of 99.57%.

This makes it highly suitable for threshold-based medical classification.

### Random Forest

Provided feature importance insights and strong interpretability.

Most Important Features:

* Worst Concave Points
* Worst Area
* Worst Radius

### ROC Threshold Optimization

Successfully reduced False Negatives by approximately 18%.

This is particularly important because undetected cancer cases can have severe medical consequences.

### Cross Validation

All models demonstrated strong stability with very low variance across folds.

---

#  Learning Outcomes

Through this project, the following concepts were learned and applied:

* End-to-End Machine Learning Pipeline Development
* Exploratory Data Analysis
* Data Preprocessing Techniques
* Feature Scaling
* Classification Algorithms
* ROC Curves and AUC
* Threshold Optimization
* Cross Validation
* Model Evaluation Metrics
* Feature Importance Analysis
* Medical Data Analytics
* Data Visualization
* Research-Oriented Problem Solving

### Technical Skills Acquired

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* Machine Learning
* Statistical Analysis
* Data Visualization

---

#  Future Enhancements

Potential improvements include:

* Deep Learning using CNNs
* Histopathology Image Classification
* Hyperparameter Optimization using GridSearchCV
* Explainable AI using SHAP
* Streamlit Web Application
* Flask-Based Deployment
* Cloud Integration
* Real-Time Clinical Dashboard

---

#  Conclusion

This project demonstrates the effectiveness of Machine Learning in supporting breast cancer diagnosis. Through comparative analysis of Logistic Regression, Support Vector Machine, and Random Forest models, highly accurate tumor classification was achieved.

The study highlights that even relatively simple machine learning algorithms can achieve outstanding performance when combined with proper preprocessing and evaluation techniques.

Furthermore, ROC Threshold Optimization significantly reduced False Negatives, making the system more suitable for healthcare applications where diagnostic accuracy is critical.

Overall, the project successfully showcases how Artificial Intelligence can contribute to early disease detection, improve healthcare decision-making, and potentially save lives through timely intervention.

---



