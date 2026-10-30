import streamlit as st

from src import theme, views
from src.predictor import load_artifacts

PAGES = ["Overview", "Forest Prediction", "Model Insights", "About the Project"]

st.set_page_config(page_title="ForestGuard AI", page_icon="🌲", layout="wide", initial_sidebar_state="collapsed")
theme.inject_css()


@st.cache_resource(show_spinner="Loading ForestGuard AI model...")
def get_model():
    """Load and cache the trained model and metadata."""
    return load_artifacts()


try:
    model, metadata = get_model()
except Exception as error:
    st.error("Unable to load the trained model. Please check the model repository and configuration.")
    st.exception(error)
    st.stop()

st.session_state.setdefault("page", PAGES[0])
st.session_state.update({k: v for k, v in views.DEFAULTS.items() if k not in st.session_state})

brand, nav = st.columns([1, 3])
brand.markdown("<div class='fg-brand'>🌲 ForestGuard <span>AI</span></div>", unsafe_allow_html=True)
nav.segmented_control("Navigation", PAGES, key="page", label_visibility="collapsed")
page = st.session_state.get("page") or PAGES[0]

if page == "Overview":
    views.overview_page(metadata)
elif page == "Forest Prediction":
    views.prediction_page(model, metadata)
elif page == "Model Insights":
    views.insights_page(metadata)
else:
    views.about_page(metadata)

st.divider()
st.caption("ForestGuard AI · UCI Covertype · Random Forest Classifier")
