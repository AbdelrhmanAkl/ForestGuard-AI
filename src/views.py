"""Page renderers for ForestGuard AI."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import insights, theme
from src.predictor import predict_cover_type
from src.utils import COVER_TYPES, get_cover_type

LINKS = {
    "Live application": "https://forestguard-ai.streamlit.app/",
    "GitHub repository": "https://github.com/AbdelrhmanAkl/ForestGuard-AI",
    "Hugging Face model": "https://huggingface.co/AbdelrahmanAkl/forestguard-ai-model",
}

# key, label, min, max, default, help. Bounds and defaults match the original app.
INPUT_GROUPS = {
    "Terrain characteristics": [
        ("Elevation", "Elevation (m)", 1859, 3858, 2500, "Height above sea level in meters."),
        ("Aspect", "Aspect (°)", 0, 360, 180, "Compass direction the slope faces, in degrees azimuth (0 = north)."),
        ("Slope", "Slope (°)", 0, 66, 15, "Steepness of the terrain in degrees."),
    ],
    "Hydrology, roads and fire points": [
        ("Horizontal_Distance_To_Hydrology", "Horizontal distance to water (m)", 0, 1397, 300, "Horizontal distance to the nearest surface water feature."),
        ("Vertical_Distance_To_Hydrology", "Vertical distance to water (m)", -173, 601, 20, "Vertical distance to the nearest surface water feature. Negative means below the water feature."),
        ("Horizontal_Distance_To_Roadways", "Horizontal distance to roads (m)", 0, 7117, 1000, "Horizontal distance to the nearest roadway."),
        ("Horizontal_Distance_To_Fire_Points", "Horizontal distance to fire points (m)", 0, 7173, 1500, "Horizontal distance to the nearest wildfire ignition point."),
    ],
    "Hillshade measurements": [
        ("Hillshade_9am", "Hillshade at 9 AM", 0, 255, 220, "Illumination index (0-255) at 9 AM on the summer solstice."),
        ("Hillshade_Noon", "Hillshade at noon", 0, 255, 230, "Illumination index (0-255) at noon on the summer solstice."),
        ("Hillshade_3pm", "Hillshade at 3 PM", 0, 255, 140, "Illumination index (0-255) at 3 PM on the summer solstice."),
    ],
}
ALL_INPUTS = [spec for group in INPUT_GROUPS.values() for spec in group]
DEFAULTS = {f"in_{s[0]}": s[4] for s in ALL_INPUTS} | {"in_wilderness": 0, "in_soil": 0}


def pct(value):
    return f"{value * 100:.2f}%"


def go_to(page):
    st.session_state["page"] = page


def build_features(values, wilderness_area, soil_type, feature_names):
    """Build the full feature dict (original order comes from metadata); one-hot encode categories."""
    features = {name: 0 for name in feature_names}
    for key, value in values.items():
        if key not in features:
            raise ValueError(f"Expected feature is missing from metadata: {key}")
        features[key] = value
    for name in (f"Wilderness_Area_{wilderness_area}", f"Soil_Type_{soil_type}"):
        if name not in features:
            raise ValueError(f"Invalid categorical feature: {name}")
        features[name] = 1
    return features


# ---------------------------------------------------------------- Overview
def overview_page(metadata):
    m = metadata["test_metrics"]
    theme.hero("Environmental intelligence", "Understand Forest Landscapes with AI",
               "Predict forest cover types using advanced machine learning and environmental data.")
    c1, c2, _ = st.columns([1, 1.3, 3])
    c1.button("Start Prediction", type="primary", on_click=go_to, args=("Forest Prediction",), width="stretch")
    c2.button("Explore Model Insights", on_click=go_to, args=("Model Insights",), width="stretch")

    theme.kpi_row([
        (pct(m["Accuracy"]), "Test accuracy"), (pct(m["Macro F1"]), "Macro F1-score"),
        (str(metadata["number_of_features"]), "Input features"), (str(metadata["number_of_classes"]), "Forest cover classes"),
    ])
    theme.section("How it works")
    theme.card_row([
        ("01", "Describe the terrain", "Enter elevation, slope, distances and hillshade values, then pick a wilderness area and soil type."),
        ("02", "Run the model", "Your inputs become the 54 features the trained Random Forest expects."),
        ("03", "Read the result", "See the predicted cover type, its confidence and the probability of all seven classes."),
    ])
    theme.section("What you can predict", "One of seven forest cover classes from the UCI Covertype dataset.")
    chips = " · ".join(f"<b>{i}</b> {c['name']}" for i, c in COVER_TYPES.items())
    st.markdown(f"<div class='fg-card'><p>{chips}</p></div>", unsafe_allow_html=True)
    st.markdown("")
    theme.section("About the data")
    st.markdown(
        f"The model was trained on the UCI Covertype dataset: **{metadata['dataset_samples']:,} records** of cartographic "
        "and environmental measurements, each labelled with its forest cover type."
    )


# -------------------------------------------------------------- Prediction
def _reset_inputs():
    st.session_state.update(DEFAULTS)
    st.session_state.pop("result", None)


def prediction_page(model, metadata):
    theme.section("Forest Prediction", "Open each group, adjust the values, then run the prediction. Defaults are a reasonable starting point.")
    for group, specs in INPUT_GROUPS.items():
        with st.expander(group, expanded=(group == "Terrain characteristics")):
            cols = st.columns(len(specs) if len(specs) <= 3 else 2)
            for n, (key, label, lo, hi, default, help_text) in enumerate(specs):
                cols[n % len(cols)].number_input(label, min_value=lo, max_value=hi, step=1,
                                                 value=st.session_state.get(f"in_{key}", default), key=f"in_{key}",
                                                 help=f"{help_text} Allowed range: {lo} to {hi}.")
    with st.expander("Wilderness area"):
        st.radio("Wilderness area", options=range(4), format_func=lambda v: f"Area {v + 1}", horizontal=True,
                 key="in_wilderness", help="Select exactly one of the four wilderness areas covered by the dataset.")
    with st.expander("Soil type"):
        st.selectbox("Soil type", options=range(40), format_func=lambda v: f"Soil type {v + 1}", key="in_soil",
                     help="Select exactly one of the 40 soil types covered by the dataset.")

    b1, b2, _ = st.columns([1.4, 1, 3])
    predict = b1.button("Predict Forest Cover", type="primary", width="stretch")
    b2.button("Reset", on_click=_reset_inputs, width="stretch")

    if predict:
        try:
            with st.spinner("Running the model..."):
                values = {s[0]: st.session_state[f"in_{s[0]}"] for s in ALL_INPUTS}
                features = build_features(values, st.session_state["in_wilderness"], st.session_state["in_soil"],
                                          metadata["feature_names"])
                if len(features) != model.n_features_in_:
                    raise ValueError("Input feature count does not match the trained model.")
                st.session_state["result"] = predict_cover_type(model, metadata, features)
        except Exception as error:
            st.session_state.pop("result", None)
            st.error(f"Prediction failed: {error}")

    result = st.session_state.get("result")
    if result is None:
        st.info("Your prediction will appear here once you click **Predict Forest Cover**.")
        return
    _render_result(result)


def _render_result(result):
    cover = get_cover_type(result["predicted_class"])
    conf = result["confidence"]
    st.markdown(
        f"<div class='fg-result'><div class='k'>Predicted forest cover · Class {result['predicted_class']}</div>"
        f"<div class='n'>{cover['name']}</div><div class='k'>Confidence {conf * 100:.2f}%</div>"
        f"<div class='fg-bar'><div style='width:{conf * 100:.1f}%'></div></div></div>",
        unsafe_allow_html=True,
    )
    ranked = sorted(result["class_probabilities"].items(), key=lambda kv: kv[1], reverse=True)
    runner_id, runner_p = ranked[1]
    closeness = "The top two classes are close, so treat this result with extra care." if conf - runner_p < 0.2 else \
        "The model strongly favors this class over the alternatives."
    st.markdown(
        f"**Interpretation.** For these inputs, the model gives the highest probability to **{cover['name']}** "
        f"({conf * 100:.1f}%). The next most likely class is {get_cover_type(runner_id)['name']} ({runner_p * 100:.1f}%). "
        f"{closeness} Confidence is the model's predicted probability, not a guarantee of correctness."
    )
    df = pd.DataFrame({
        "Class": [f"{cid} · {get_cover_type(cid)['name']}" for cid in map(int, result["class_probabilities"])],
        "Probability": [p * 100 for p in result["class_probabilities"].values()],
    })
    df["Predicted"] = df["Class"].str.startswith(f"{result['predicted_class']} ·")
    fig = px.bar(df.sort_values("Probability"), x="Probability", y="Class", orientation="h", color="Predicted",
                 color_discrete_map={True: theme.GREEN, False: theme.SAGE},
                 text=df.sort_values("Probability")["Probability"].map("{:.2f}%".format))
    fig.update_xaxes(title="Probability (%)", range=[0, 105])
    fig.update_yaxes(title=None)
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(theme.style_figure(fig, 380), width="stretch")


# ---------------------------------------------------------------- Insights
def insights_page(metadata):
    theme.section("Model Insights", "Evaluation results from the project's saved reports. Nothing here is recomputed or modified.")
    tabs = st.tabs(["Performance", "Model comparison", "Feature importance", "Confusion matrix"])
    with tabs[0]:
        m = metadata["test_metrics"]
        theme.kpi_row([(pct(m[k]), k) for k in ("Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1")])
        report = insights.load_report("classification_report.csv")
        if report is None:
            st.info("The classification report is not available.")
        else:
            st.markdown("#### F1-score by class")
            fig, _ = insights.class_performance_chart(report)
            st.plotly_chart(fig, width="stretch")
    with tabs[1]:
        comparison = insights.load_report("model_comparison.csv")
        if comparison is None:
            st.info("The model comparison report is not available.")
        else:
            st.caption("Validation results for each candidate model. The y-axis starts at 0.80 to make differences visible.")
            st.plotly_chart(insights.comparison_chart(comparison), width="stretch")
            st.dataframe(comparison, hide_index=True, width="stretch")
    with tabs[2]:
        importance = insights.load_report("feature_importance.csv")
        if importance is None:
            st.info("The feature importance report is not available.")
        else:
            st.caption("How much the trained model relies on each feature. This does not establish causation.")
            st.plotly_chart(insights.importance_chart(importance), width="stretch")
    with tabs[3]:
        matrix = insights.load_report("test_confusion_matrix.csv")
        if matrix is None:
            st.info("The confusion matrix is not available.")
        else:
            normalize = st.toggle("Show as % of each actual class", value=False)
            st.plotly_chart(insights.confusion_chart(matrix, normalize), width="stretch")


# ------------------------------------------------------------------- About
def about_page(metadata):
    theme.section("About the Project")
    st.markdown(
        "ForestGuard AI is an end-to-end machine learning application that predicts forest cover type from "
        "cartographic and environmental features."
    )
    theme.card_row([
        ("", "Dataset", f"UCI Covertype: {metadata['dataset_samples']:,} records, {metadata['number_of_features']} features, "
                        f"{metadata['number_of_classes']} cover classes."),
        ("", "Methodology", "A tuned Random Forest classifier, compared against Decision Tree, XGBoost and other Random Forest variants, "
                            "and evaluated on a held-out test set."),
        ("", "Technology", "Python, scikit-learn, pandas, Plotly and Streamlit. The trained model is hosted on Hugging Face."),
    ])
    theme.section("Limitations")
    st.markdown(
        "- Predictions are statistical estimates from 54 cartographic features, not field surveys.\n"
        "- The model learned from one geographic region (the Covertype dataset), so it may not generalize elsewhere.\n"
        "- Minority classes such as Cottonwood/Willow and Aspen have lower recall than the majority classes."
    )
    theme.section("Links")
    cols = st.columns(3)
    for col, (label, url) in zip(cols, LINKS.items()):
        col.link_button(label, url, width="stretch")
    st.markdown("")
    st.markdown("<p class='fg-muted'>Built by <b>Abdelrahman Akl</b> · "
                "<a href='https://github.com/AbdelrhmanAkl'>GitHub</a> · "
                "<a href='https://huggingface.co/AbdelrahmanAkl'>Hugging Face</a></p>", unsafe_allow_html=True)
