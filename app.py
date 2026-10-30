
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predictor import load_artifacts, predict_cover_type
from src.utils import get_cover_type


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
VISUALIZATIONS_DIR = PROJECT_ROOT / "outputs" / "visualizations"

FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_IMAGE = VISUALIZATIONS_DIR / "feature_importance.png"
CONFUSION_MATRIX_IMAGE = VISUALIZATIONS_DIR / "test_confusion_matrix.png"


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ForestGuard AI",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Model Loading
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading ForestGuard AI model...")
def get_model():
    """Load and cache the trained model and metadata."""
    return load_artifacts()


try:
    model, metadata = get_model()

except Exception as error:
    st.error(
        "Unable to load the trained model. "
        "Please check the model repository and configuration."
    )
    st.exception(error)
    st.stop()


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

def build_features(
    elevation,
    aspect,
    slope,
    horizontal_hydrology,
    vertical_hydrology,
    horizontal_roadways,
    hillshade_9am,
    hillshade_noon,
    hillshade_3pm,
    horizontal_fire,
    wilderness_area,
    soil_type,
    feature_names,
):
    """
    Build the complete 54-feature input.

    Feature order follows the original training metadata.
    Categorical features use one-hot encoding.
    """

    features = {name: 0 for name in feature_names}

    numerical_features = {
        "Elevation": elevation,
        "Aspect": aspect,
        "Slope": slope,
        "Horizontal_Distance_To_Hydrology": horizontal_hydrology,
        "Vertical_Distance_To_Hydrology": vertical_hydrology,
        "Horizontal_Distance_To_Roadways": horizontal_roadways,
        "Hillshade_9am": hillshade_9am,
        "Hillshade_Noon": hillshade_noon,
        "Hillshade_3pm": hillshade_3pm,
        "Horizontal_Distance_To_Fire_Points": horizontal_fire,
    }

    for feature_name, value in numerical_features.items():
        if feature_name not in features:
            raise ValueError(
                f"Expected feature is missing from metadata: {feature_name}"
            )

        features[feature_name] = value

    wilderness_feature = f"Wilderness_Area_{wilderness_area}"
    soil_feature = f"Soil_Type_{soil_type}"

    if wilderness_feature not in features:
        raise ValueError(
            f"Invalid wilderness feature: {wilderness_feature}"
        )

    if soil_feature not in features:
        raise ValueError(
            f"Invalid soil feature: {soil_feature}"
        )

    features[wilderness_feature] = 1
    features[soil_feature] = 1

    return features


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("ForestGuard AI")

st.subheader(
    "Multi-Class Forest Cover Type Classification"
)

st.markdown(
    """
    Predict forest cover types using geographical and environmental
    characteristics with a trained Random Forest classification model.
    """
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("ForestGuard AI")

    st.caption("Machine Learning Prediction System")

    st.divider()

    st.markdown("### Model Information")

    st.write(f"**Model:** {metadata['model_type']}")
    st.write(f"**Features:** {metadata['number_of_features']}")
    st.write(f"**Classes:** {metadata['number_of_classes']}")
    st.write(f"**Dataset:** {metadata['dataset']}")

    st.divider()

    st.markdown("### Test Performance")

    test_metrics = metadata["test_metrics"]

    sidebar_metrics = [
        ("Test Accuracy", "Accuracy"),
        ("Macro F1-Score", "Macro F1"),
        ("Weighted F1-Score", "Weighted F1"),
    ]

    for label, metric_key in sidebar_metrics:
        st.metric(
            label,
            f"{test_metrics[metric_key] * 100:.2f}%",
        )

    st.divider()

    st.caption(
        "Model evaluation is based on an untouched test dataset."
    )


# --------------------------------------------------
# Input Form
# --------------------------------------------------

st.header("Forest Environment Configuration")

st.info(
    "Enter the environmental characteristics and select one wilderness "
    "area and one soil type to generate a prediction."
)

with st.form("prediction_form"):

    st.subheader("1. Terrain and Geographic Features")

    col1, col2, col3 = st.columns(3)

    with col1:

        elevation = st.number_input(
            "Elevation (meters)",
            min_value=1859,
            max_value=3858,
            value=2500,
            step=1,
        )

        aspect = st.number_input(
            "Aspect (degrees)",
            min_value=0,
            max_value=360,
            value=180,
            step=1,
        )

        slope = st.number_input(
            "Slope (degrees)",
            min_value=0,
            max_value=66,
            value=15,
            step=1,
        )

    with col2:

        horizontal_hydrology = st.number_input(
            "Horizontal Distance to Hydrology",
            min_value=0,
            max_value=1397,
            value=300,
            step=1,
        )

        vertical_hydrology = st.number_input(
            "Vertical Distance to Hydrology",
            min_value=-173,
            max_value=601,
            value=20,
            step=1,
        )

        horizontal_roadways = st.number_input(
            "Horizontal Distance to Roadways",
            min_value=0,
            max_value=7117,
            value=1000,
            step=1,
        )

    with col3:

        hillshade_9am = st.number_input(
            "Hillshade at 9 AM",
            min_value=0,
            max_value=255,
            value=220,
            step=1,
        )

        hillshade_noon = st.number_input(
            "Hillshade at Noon",
            min_value=0,
            max_value=255,
            value=230,
            step=1,
        )

        hillshade_3pm = st.number_input(
            "Hillshade at 3 PM",
            min_value=0,
            max_value=255,
            value=140,
            step=1,
        )

    horizontal_fire = st.number_input(
        "Horizontal Distance to Fire Points",
        min_value=0,
        max_value=7173,
        value=1500,
        step=1,
    )

    st.divider()

    st.subheader("2. Wilderness Area")

    wilderness_area = st.selectbox(
        "Select Wilderness Area",
        options=list(range(4)),
        format_func=lambda value: f"Wilderness Area {value + 1}",
    )

    st.divider()

    st.subheader("3. Soil Type")

    soil_type = st.selectbox(
        "Select Soil Type",
        options=list(range(40)),
        format_func=lambda value: f"Soil Type {value + 1}",
    )

    submitted = st.form_submit_button(
        "Predict Forest Cover Type",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Prediction Results
# --------------------------------------------------

if submitted:

    try:

        input_features = build_features(
            elevation=elevation,
            aspect=aspect,
            slope=slope,
            horizontal_hydrology=horizontal_hydrology,
            vertical_hydrology=vertical_hydrology,
            horizontal_roadways=horizontal_roadways,
            hillshade_9am=hillshade_9am,
            hillshade_noon=hillshade_noon,
            hillshade_3pm=hillshade_3pm,
            horizontal_fire=horizontal_fire,
            wilderness_area=wilderness_area,
            soil_type=soil_type,
            feature_names=metadata["feature_names"],
        )

        if len(input_features) != model.n_features_in_:
            raise ValueError(
                "Input feature count does not match the trained model."
            )

        result = predict_cover_type(
            model,
            metadata,
            input_features,
        )

        cover_info = get_cover_type(
            result["predicted_class"]
        )

        st.divider()

        st.header("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Predicted Forest Cover",
                cover_info["name"],
            )

        with result_col2:

            st.metric(
                "Prediction Confidence",
                f"{result['confidence'] * 100:.2f}%",
            )

        st.progress(result["confidence"])

        st.caption(
            "Confidence represents the model's predicted probability "
            "for the selected class. It is not a guarantee of correctness."
        )

        st.markdown(
            f"**Description:** {cover_info['description']}"
        )

        # Class probabilities
        st.subheader("Class Probability Distribution")

        probability_df = pd.DataFrame(
            [
                {
                    "Cover Type": get_cover_type(
                        int(class_id)
                    )["name"],
                    "Probability": probability * 100,
                }
                for class_id, probability
                in result["class_probabilities"].items()
            ]
        )

        probability_df = probability_df.sort_values(
            "Probability",
            ascending=True,
        )

        probability_chart = px.bar(
            probability_df,
            x="Probability",
            y="Cover Type",
            orientation="h",
            text=probability_df["Probability"].map(
                lambda value: f"{value:.2f}%"
            ),
            labels={
                "Probability": "Predicted Probability (%)",
                "Cover Type": "Forest Cover Type",
            },
            title="Probability of Each Forest Cover Type",
        )

        probability_chart.update_layout(
            height=420,
            yaxis={"categoryorder": "total ascending"},
        )

        st.plotly_chart(
            probability_chart,
            use_container_width=True,
        )

        with st.expander(
            "View Detailed Prediction Probabilities"
        ):

            st.dataframe(
                probability_df.sort_values(
                    "Probability",
                    ascending=False,
                ),
                use_container_width=True,
                hide_index=True,
            )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------

st.divider()

st.header("Model Evaluation")

test_metrics = metadata["test_metrics"]

metric_columns = st.columns(4)

evaluation_metrics = [
    ("Accuracy", "Accuracy"),
    ("Macro Precision", "Macro Precision"),
    ("Macro Recall", "Macro Recall"),
    ("Macro F1-Score", "Macro F1"),
]

for column, (label, metric_key) in zip(
    metric_columns,
    evaluation_metrics,
):

    column.metric(
        label,
        f"{test_metrics[metric_key] * 100:.2f}%",
    )


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

st.divider()

st.header("Feature Importance")

if FEATURE_IMPORTANCE_PATH.exists():

    try:

        importance_df = pd.read_csv(
            FEATURE_IMPORTANCE_PATH
        )

        st.caption(
            "Feature importance shows how much the trained model relies "
            "on each feature. It does not establish causation."
        )

        st.dataframe(
            importance_df.head(10),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as error:

        st.warning(
            f"Unable to load feature importance report: {error}"
        )

else:

    st.info(
        "Feature importance report is not available."
    )


# --------------------------------------------------
# Model Visualizations
# --------------------------------------------------

st.divider()

st.header("Model Visualizations")

visual_col1, visual_col2 = st.columns(2)

with visual_col1:

    st.subheader("Feature Importance")

    if FEATURE_IMPORTANCE_IMAGE.exists():

        st.image(
            str(FEATURE_IMPORTANCE_IMAGE),
            use_container_width=True,
        )

    else:

        st.info(
            "Feature importance visualization is not available."
        )


with visual_col2:

    st.subheader("Test Confusion Matrix")

    if CONFUSION_MATRIX_IMAGE.exists():

        st.image(
            str(CONFUSION_MATRIX_IMAGE),
            use_container_width=True,
        )

    else:

        st.info(
            "Confusion matrix visualization is not available."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "ForestGuard AI | UCI Covertype Dataset | Random Forest Classifier"
)
