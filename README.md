# ForestGuard AI — Forest Cover Type Classification

An end-to-end Machine Learning project that predicts forest cover types from cartographic and environmental features using an optimized Random Forest classifier.

The system classifies forest areas into **7 forest cover categories** using the UCI Covertype dataset, achieving **96.20% test accuracy** and **93.29% Macro F1-score**.

<p align="center">
  <a href="https://forestguard-ai.streamlit.app/"><strong>Live Demo</strong></a> ·
  <a href="https://huggingface.co/AbdelrahmanAkl/forestguard-ai-model"><strong>Hugging Face Model</strong></a> ·
  <a href="https://github.com/AbdelrhmanAkl/ForestGuard-AI"><strong>GitHub Repository</strong></a>
</p>

---

## Overview

ForestGuard AI is a multi-class classification system designed to predict forest cover types based on geographical and environmental characteristics.

It uses 54 cartographic features, including elevation, slope, hydrological distances, hillshade measurements, wilderness areas, and soil types.

The project covers the complete machine learning workflow:

- Data exploration and quality analysis.
- Data preprocessing and stratified splitting.
- Training and comparing multiple classification algorithms.
- Hyperparameter optimization using RandomizedSearchCV.
- Final evaluation on an untouched test set.
- Model serialization and artifact hosting.
- Interactive prediction through a Streamlit web application.

## Live Application

Try the deployed application:

**[Launch ForestGuard AI](https://forestguard-ai.streamlit.app/)**

The application provides:
- Interactive terrain feature inputs.
- Predicted forest cover category.
- Prediction confidence.
- Class probability visualization.
- Model evaluation metrics.
- Feature importance visualization.

## Model Performance

The final optimized Random Forest model was evaluated on an untouched test set.

| Metric | Test Score |
|---|---:|
| Accuracy | 96.20% |
| Macro Precision | 94.73% |
| Macro Recall | 92.00% |
| Macro F1-score | 93.29% |
| Weighted F1-score | 96.18% |

### Model Comparison

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

Best parameters:

```python
{
    "n_estimators": 100,
    "max_depth": 30,
    "max_features": 0.7,
    "min_samples_split": 5,
    "min_samples_leaf": 1
}
```

Best cross-validation Macro F1-score: **0.9243**

## Dataset

**UCI Covertype Dataset**

- Total records: 581,012
- Input features: 54
- Target classes: 7
- Task: Multi-class classification

Dataset source:

[UCI Machine Learning Repository — Covertype](https://archive.ics.uci.edu/dataset/31/covertype)

The dataset contains cartographic variables describing forest areas, including:

- Elevation
- Aspect
- Slope
- Horizontal and vertical distance to hydrology
- Horizontal distance to roadways
- Hillshade measurements
- Wilderness area indicators
- Soil type indicators

## Forest Cover Categories

The model predicts one of the following seven classes:

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
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Model Optimization | RandomizedSearchCV |
| Visualization | Matplotlib, Plotly |
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

The trained model is hosted separately on Hugging Face to keep the GitHub repository lightweight.

**Model artifact:** [forestguard-ai-model](https://huggingface.co/AbdelrahmanAkl/forestguard-ai-model)

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/AbdelrhmanAkl/ForestGuard-AI.git
cd ForestGuard-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

**Model download:** If the trained model is not available locally, the application automatically downloads it from the public Hugging Face model repository.

## Model Evaluation

The project includes evaluation reports and visualizations:

- Classification report.
- Test confusion matrix.
- Feature importance analysis.
- Model comparison results.
- Final test metrics.

The feature importance analysis identifies which input features the trained Random Forest relies on most. These values represent model importance, not causal relationships.

## Limitations

- Predictions are based on the UCI Covertype dataset and its original feature definitions.
- The model is not a substitute for field surveys or professional forestry assessment.
- Performance on geographical regions or data distributions outside the original dataset has not been independently established.
- Prediction confidence represents the model's estimated class probabilities, not a guarantee of correctness.

## Author

**Abdelrahman Akl**

AI Engineer | Machine Learning | Deep Learning | Agentic AI

- GitHub: [@AbdelrhmanAkl](https://github.com/AbdelrhmanAkl)
- Hugging Face: [@AbdelrahmanAkl](https://huggingface.co/AbdelrahmanAkl)
- Live Project: [ForestGuard AI](https://forestguard-ai.streamlit.app/)

---

<p align="center">
  Built with Python and Machine Learning.
</p>