# ForestGuard AI — Forest Cover Type Classification

An end-to-end Machine Learning project that predicts forest cover types from cartographic and environmental features using an optimized Random Forest classifier.

ForestGuard AI classifies forest areas into **7 forest cover categories** using the UCI Covertype dataset, achieving **96.20% test accuracy** and a **93.29% Macro F1-score**.

<p align="center">
  <a href="https://forestguard-ai.streamlit.app/"><strong>Live Demo</strong></a>
  &nbsp;|&nbsp;
  <a href="https://huggingface.co/AbdelrahmanAkl/forestguard-ai-model"><strong>Hugging Face Model</strong></a>
  &nbsp;|&nbsp;
  <a href="https://github.com/AbdelrhmanAkl/ForestGuard-AI"><strong>GitHub Repository</strong></a>
</p>

---

## Overview

ForestGuard AI is a multi-class classification system designed to predict forest cover types based on geographical and environmental characteristics.

The model uses 54 cartographic features, including elevation, slope, hydrological distances, hillshade measurements, wilderness areas, and soil types.

The project demonstrates a complete Machine Learning workflow:

- Exploratory Data Analysis (EDA) and data quality analysis.
- Data preprocessing and stratified splitting.
- Training and comparison of multiple classification algorithms.
- Hyperparameter optimization using RandomizedSearchCV.
- Final evaluation on an untouched test set.
- Model serialization and hosting on Hugging Face.
- Interactive predictions through a Streamlit web application.

## Live Application

<p align="center">
  <a href="https://forestguard-ai.streamlit.app/">
    <strong>Launch ForestGuard AI</strong>
  </a>
</p>

The interactive application provides:

- Terrain feature inputs.
- Predicted forest cover category.
- Prediction confidence.
- Class probability visualization.
- Model evaluation metrics.
- Feature importance visualization.

## Model Performance

The final optimized Random Forest model was evaluated on an untouched test set containing 116,203 samples.

| Evaluation Metric | Test Score |
|---|---:|
| Accuracy | **96.20%** |
| Macro Precision | 94.73% |
| Macro Recall | 92.00% |
| Macro F1-score | **93.29%** |
| Weighted F1-score | 96.18% |

### Model Comparison

The following models were compared using validation performance:

| Model | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| Decision Tree | 93.13% | 88.52% |
| Random Forest | 94.89% | 91.57% |
| XGBoost | 87.75% | 86.07% |
| Balanced Random Forest | 95.00% | 91.41% |
| Tuned Random Forest | **96.21%** | **93.24%** |

The tuned Random Forest achieved the strongest validation Macro F1-score among the evaluated models.

### Hyperparameter Optimization

RandomizedSearchCV was used with 8 parameter combinations and 3-fold stratified cross-validation, optimizing Macro F1.

**Best hyperparameters:**

```python
{
    "n_estimators": 100,
    "max_depth": 30,
    "max_features": 0.7,
    "min_samples_split": 5,
    "min_samples_leaf": 1
}
```

- Best cross-validation Macro F1-score: **0.9243**
- Search candidates: 8
- Cross-validation folds: 3
- Optimization metric: Macro F1

## Dataset

**UCI Covertype Dataset**

Source: [UCI Machine Learning Repository — Covertype](https://archive.ics.uci.edu/dataset/31/covertype)

| Dataset Property | Value |
|---|---:|
| Total Records | 581,012 |
| Input Features | 54 |
| Target Classes | 7 |
| Task | Multi-class Classification |

The dataset contains cartographic and environmental variables describing forest areas.

### Feature Categories

The 54 input features include:

- Elevation
- Aspect
- Slope
- Horizontal and vertical distance to hydrology
- Horizontal distance to roadways
- Horizontal distance to fire points
- Hillshade measurements
- Wilderness area indicators
- Soil type indicators

## Forest Cover Categories

The model predicts one of the following seven forest cover types:

| Class | Forest Cover Type |
|---:|---|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Model Optimization | RandomizedSearchCV |
| Data Visualization | Matplotlib, Plotly |
| Web Application | Streamlit |
| Model Serialization | Joblib |
| Model Hosting | Hugging Face Hub |
| Version Control | Git, GitHub |

## Project Structure

```text
ForestGuard-AI/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── data/
│
├── models/
│   └── model_metadata.json
│
├── notebooks/
│   └── Forest_Cover_Type_Classification.ipynb
│
├── outputs/
│   ├── reports/
│   │   ├── classification_report.csv
│   │   ├── feature_importance.csv
│   │   ├── model_comparison.csv
│   │   ├── test_confusion_matrix.csv
│   │   └── test_metrics.csv
│   │
│   └── visualizations/
│       ├── feature_importance.png
│       └── test_confusion_matrix.png
│
└── src/
    ├── __init__.py
    ├── predictor.py
    └── utils.py
```

**Note:** The trained model artifact is hosted separately on Hugging Face to keep the GitHub repository lightweight.

[View ForestGuard AI Model on Hugging Face](https://huggingface.co/AbdelrahmanAkl/forestguard-ai-model)

## Installation and Local Setup

Follow these steps to run ForestGuard AI locally.

### 1. Clone the Repository

```bash
git clone https://github.com/AbdelrhmanAkl/ForestGuard-AI.git
cd ForestGuard-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

### Model Download

If the trained model is not available locally, the application automatically downloads the model artifact from the public Hugging Face repository.

## Model Evaluation and Visualizations

The project includes detailed evaluation reports and visualizations:

- Classification report.
- Test confusion matrix.
- Feature importance analysis.
- Model comparison results.
- Final test metrics.

Feature importance indicates which input features the trained Random Forest relies on most. These values represent model reliance, not causal relationships.

## Limitations

- Predictions are based on the UCI Covertype dataset and its original feature definitions.
- The model is not a substitute for field surveys or professional forestry assessment.
- Performance on geographical regions or data distributions outside the original dataset has not been independently established.
- Prediction confidence represents the model's estimated class probabilities and does not guarantee correctness.

## Author

**Abdelrahman Akl**

AI Engineer | Machine Learning | Deep Learning | Agentic AI

- **GitHub:** [@AbdelrhmanAkl](https://github.com/AbdelrhmanAkl)
- **Hugging Face:** [@AbdelrahmanAkl](https://huggingface.co/AbdelrahmanAkl)
- **Live Project:** [ForestGuard AI](https://forestguard-ai.streamlit.app/)

---

<p align="center">
  Built with Python and Machine Learning.
</p>