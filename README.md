#  Breast Cancer Detection using Machine Learning

> **Gaurav Sharma | 23BDA70050 | Chandigarh University**  
> Python · Scikit-learn · Pandas · Matplotlib · 2025

---

##  About the Project

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

##  Models Used

I trained and compared 3 models:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Neg. |
|-------|----------|-----------|--------|----------|---------|-----------|
| Logistic Regression | **98.25%** | 98.61% | 98.61% | 98.61% | 99.54% | 1 |
| SVM (RBF Kernel) | 97.37% | 98.59% | 97.22% | 97.90% | **99.57%** | 2 |
| Random Forest | 95.61% | 95.89% | 97.22% | 96.55% | 99.32% | 3 |
| RF (ROC-Optimised) | 96.49% | — | Higher | — | — | ↓ ~18% FN |

---

##  Project Structure

```
Breast-Cancer-Detection/
│
├── breast_cancer_detection.py   ← Main ML pipeline (run this)
│
├── charts/
│   ├── 01_eda_overview.png          ← Class distribution, correlation, boxplot
│   ├── 02_model_comparison.png      ← Accuracy bars + ROC curves
│   ├── 03_confusion_matrices.png    ← All 3 model confusion matrices
│   ├── 04_feature_importance.png    ← Top 15 RF feature importances
│   └── 05_cv_scores.png             ← 5-Fold cross-validation results
│
├── Project_Report.docx          ← Full 30-page project report
├── Breast_Cancer_Detection.pptx ← 15-slide presentation
└── README.md
```

---

## ▶ How to Run

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

##  Output Charts

The script automatically generates these 5 charts:

1. **EDA Overview** — class balance bar chart, feature correlation with target, mean radius boxplot by class
2. **Model Comparison** — grouped bar chart (Acc/Prec/Recall/F1) + ROC curves for all 3 models
3. **Confusion Matrices** — heatmaps showing TP, TN, FP, FN for each model
4. **Feature Importance** — top 15 most important features from Random Forest
5. **Cross-Validation** — boxplot of 5-fold accuracy scores per model

---

##  Key Findings

- **Logistic Regression** gave the best accuracy of **98.25%** — even though it's the simplest model. This shows the dataset is largely linearly separable after scaling.
- **SVM** achieved the highest ROC-AUC of **99.57%**, making it best for threshold-based clinical tuning.
- **Random Forest** provides feature importances — the most useful features are *worst concave points*, *worst area*, and *worst radius*.
- **ROC-threshold tuning** (Youden's J statistic) on Random Forest reduced false negatives by **~18%**, which is important in medical settings where missing a cancer case is dangerous.
- **5-Fold Cross-Validation** shows all models generalise well with standard deviation < 0.01.

---

##  Technologies Used

- **Python 3.x**
- **scikit-learn** — models, preprocessing, metrics
- **pandas** — data loading, manipulation
- **matplotlib + seaborn** — all visualisations
- **numpy** — numerical operations

---

##  What I Learned

- How to build a full ML pipeline from scratch
- Why StandardScaler is needed for SVM and Logistic Regression
- How ROC curves and AUC work and why they matter in medical ML
- How to reduce false negatives using threshold tuning
- How Random Forest feature importance helps understand the data

---

##  Future Improvements

- Try deep learning (CNN) on histopathology slide images
- Hyperparameter tuning using GridSearchCV or Optuna
- Deploy as a web app using Flask or Streamlit
- Add SHAP explainability for individual predictions

---

##  Author

**Gaurav Sharma**  
Roll No: 23BDA70050  
B.E (Data Science / CSE) — Chandigarh University  
GitHub: [@GauravSharma1534](https://github.com/GauravSharma1534)

---


